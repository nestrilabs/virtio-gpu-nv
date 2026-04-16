// SPDX-License-Identifier: GPL-2.0
/*
 * virtio-gpu-nv: NVIDIA GPU ioctl proxy for libkrun VMs.
 *
 * Each guest open("/dev/nvidia*") creates a new host FD via the VMM.
 * Ioctls are forwarded over the control virtqueue; mmap requests result
 * in KVM memory slots set up by the VMM so hot-path GPU writes go direct
 * through EPT — no VMM involvement in the render loop.
 *
 * Guest kernel driver — runs inside the VM.
 * Place in: drivers/virtio/virtio_gpu_nv.c (libkrunfw tree)
 */

#include <drm/drm.h>
#include <linux/cdev.h>
#include <linux/completion.h>
#include <linux/cpu.h>
#include <linux/file.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/pci-ecam.h>
#include <linux/pci.h>
#include <linux/proc_fs.h>
#include <linux/scatterlist.h>
#include <linux/seq_file.h>
#include <linux/slab.h>
#include <linux/topology.h>
#include <linux/uaccess.h>
#include <linux/virtio.h>
#include <linux/virtio_config.h>
#include <linux/virtio_ids.h>

#include "gen/nvgpu_rmalloc_classes.h"
#include "gen/nvgpu_v1v2_rewrites.h"
#include "nvgpu_rm_intercepts.h"

/* module_kset is exported from kernel/module/sysfs.c */
extern struct kset *module_kset;

/* ───────── virtio device identity ───────── */

#define VIRTIO_ID_GPU_NV 45

/* Feature bits */
#define VIRTIO_GPU_NV_F_UVM 0
#define VIRTIO_GPU_NV_F_ENCODE 1
#define VIRTIO_GPU_NV_F_GRAPHICS 2

/* ───────── NVIDIA device node numbers ───────── */

#define NV_MAJOR 195
#define NV_CTL_MINOR 255
#define NV_UVM_MAJOR 237
#define NV_CAPS_MAJOR 240

/* ───────── Wire protocol constants ───────── */

#define NVGPU_MSG_OPEN 1
#define NVGPU_MSG_CLOSE 2
#define NVGPU_MSG_IOCTL 3
#define NVGPU_MSG_MMAP 4
#define NVGPU_MSG_MUNMAP 5
#define NVGPU_MSG_GET_PROC_FILES 6
#define NVGPU_MSG_GET_SYS_FILES 7

/* device_type values for OPEN */
#define NVGPU_DEV_CTL 255
#define NVGPU_DEV_UVM 256
#define NVGPU_DEV_UVM_TOOLS 257
#define NV_MODESET_MINOR 254
#define NVGPU_DEV_MODESET 258

/* capability bits */
#define NVGPU_CAP_COMPUTE (1 << 0)
#define NVGPU_CAP_GRAPHICS (1 << 1)
#define NVGPU_CAP_VIDEO (1 << 2)
#define NVGPU_CAP_UTILITY (1 << 3)

/* NVIDIA ioctl numbers that require nested-pointer marshalling */
#define NV_ESC_RM_CONTROL 0x2a
#define NV_ESC_RM_ALLOC 0x2b
/* UVM_INITIALIZE ioctl nr */
#define UVM_INITIALIZE_NR 0x30

#define NVGPU_DEV_DRI_BASE 512

/* ───────── Wire protocol structs ───────── */

struct nvgpu_msg_hdr {
  __le32 msg_type;
  __le32 handle;
  __le32 status;
  __le32 padding;
} __packed;

struct nvgpu_open_req {
  struct nvgpu_msg_hdr hdr;
  __le32 device_type;
  __le32 flags;
} __packed;

struct nvgpu_open_resp {
  struct nvgpu_msg_hdr hdr;
} __packed;

struct nvgpu_ioctl_req {
  struct nvgpu_msg_hdr hdr;
  __le32 cmd;
  __le32 data_len;
  __le32 nested_offset;
  __le32 nested_len;
  /* followed by: data_len bytes top-level struct,
   *              nested_len bytes nested data       */
} __packed;

struct nvgpu_ioctl_resp {
  struct nvgpu_msg_hdr hdr;
  __le32 data_len;
  __le32 nested_len;
  /* followed by: data_len bytes modified top-level,
   *              nested_len bytes modified nested   */
} __packed;

struct nvgpu_mmap_req {
  struct nvgpu_msg_hdr hdr;
  __le64 size;
  __le64 offset;
  __le32 prot;
  __le32 padding;
} __packed;

struct nvgpu_mmap_resp {
  struct nvgpu_msg_hdr hdr;
  __le64 guest_phys_addr;
  __le64 size;
  __le32 mapping_id;
  __le32 padding;
} __packed;

struct nvgpu_munmap_req {
  struct nvgpu_msg_hdr hdr;
  __le32 mapping_id;
  __le32 padding;
} __packed;

struct nvgpu_munmap_resp {
  struct nvgpu_msg_hdr hdr;
} __packed;

/* VMM response: stream of nvgpu_proc_file_entry records,
 * terminated by an entry with path_len == 0 */
struct nvgpu_proc_file_entry {
  __le32 path_len;    /* bytes in path[], 0 = end of stream */
  __le32 content_len; /* bytes in content[] */
  /* followed by: path_len bytes of path (no NUL),
   *              content_len bytes of content          */
} __packed;

/* Per-GPU slot in VMM config space — 1088 bytes */
struct virtio_gpu_nv_gpu_slot {
  char pci_addr[16];    /*    0.. 16  directory name          */
  __le32 minor;         /*   16.. 20  /dev/nvidia<minor>      */
  __le32 info_len;      /*   20.. 24  valid bytes in info_text */
  __le32 padding[1];    /*   24.. 28                          */
  char info_text[1060]; /*   28..1088 raw information content  */
} __packed;             /* 1088 bytes */

struct nvgpu_fd_translation_entry {
  __le32 nr;
  __le32 payload_offset;
} __packed;

/* VMM config space layout */
struct virtio_gpu_nv_config {
  char driver_version[32];               /* 0.. 32  */
  __le32 num_gpus;                       /* 32.. 36 */
  __le32 caps;                           /* 36.. 40 */
  __le32 gpu_device_ids[8];              /* 40.. 72 */
  struct virtio_gpu_nv_gpu_slot gpus[8]; /* 72..    */
  __le32 num_fd_translations;
  __le32 _pad;
  struct nvgpu_fd_translation_entry fd_translations[16];
} __packed;

static_assert(sizeof(struct virtio_gpu_nv_gpu_slot) == 1088,
              "gpu_slot size mismatch");
static_assert(sizeof(struct virtio_gpu_nv_config) == 8912,
              "virtio_gpu_nv_config size mismatch with VMM");
static_assert(offsetof(struct virtio_gpu_nv_config, num_fd_translations) ==
                  8776,
              "fd_translations offset mismatch with VMM");

/* ───────── NVIDIA ioctl parameter structs ───────── */

struct NVOS54_PARAMETERS {
  __le32 hClient;
  __le32 hObject;
  __le32 cmd;
  __le32 flags;
  __le64 params; /* pointer to sub-command data in guest VA */
  __le32 paramsSize;
  __le32 status;
} __packed;

struct NVOS64_PARAMETERS {
  __le32 hRoot;
  __le32 hObjectParent;
  __le32 hObjectNew;
  __le32 hClass;
  __le64 pAllocParms;      /* pointer to class-specific alloc params */
  __le64 pRightsRequested; /* usually NULL */
  __le32 paramsSize;
  __le32 flags;
  __le32 status;
} __packed;

/* ───────── Driver state ───────── */

struct nvgpu_device;

struct nvgpu_dri_dev {
  char name[32];
  u32 major;
  u32 minor;
  u32 gpu_id;
  struct cdev cdev;
  bool registered;
  u32 index;
  struct nvgpu_device *dev;
  /* sysfs drm tree under the PCI device — required by Vulkan ICD */
  struct kobject *drm_kobj;      /* .../pci_addr/drm          */
  struct kobject *drm_node_kobj; /* .../pci_addr/drm/<name>   */
};

struct nvgpu_numa_attr {
  struct kobj_attribute kattr;
  struct nvgpu_device *dev;
  char *(*get_buf)(struct nvgpu_device *);
};

/* ── Fake PCI device state ── */
struct nvgpu_pci_slot {
  char pci_addr[16]; /* "0000:08:00.0"      */
  u8 config[256];    /* raw config space    */
  bool config_valid;
  u16 domain;
  u8 bus_nr;
  u8 slot;
  u8 func;
};

struct nvgpu_pci_root {
  int domain; /* MUST be first — x86 pci_domain_nr()
               * reads domain from sysdata offset 0 */
  struct nvgpu_pci_slot slot;
  struct nvgpu_device *nvdev; /* back pointer        */
  struct pci_host_bridge *bridge;
  struct pci_dev *pdev; /* first (only) device on this bus */
  bool registered;
};

struct nvgpu_device {
  struct virtio_device *vdev;
  struct virtqueue *ctrl_vq;
  struct virtqueue *event_vq;

  /* Character device registration */
  struct cdev cdev_gpu[248]; /* /dev/nvidia0 … nvidia247 */
  struct cdev cdev_ctl;      /* /dev/nvidiactl            */
  struct cdev cdev_uvm;      /* /dev/nvidia-uvm           */
  dev_t uvm_devno;           /* dynamic major for UVM     */
  struct cdev cdev_caps;     /* /dev/nvidia-caps */
  dev_t caps_devno;          /* dynamic major for nvidia-caps */
  struct cdev cdev_modeset;  /* /dev/nvidia-modeset */
  dev_t modeset_devno;

  /* Config read from VMM */
  char driver_version[32];
  u32 num_gpus;
  u32 caps;

  /* Serialise virtqueue access */
  struct mutex vq_lock;

  /* Completion for synchronous request */
  struct completion req_done;
  void *resp_buf;
  int resp_len;

  /* GPU slots read from config space at probe */
  struct virtio_gpu_nv_gpu_slot gpu_slots[8];

  /* FD translation table received from backend */
  struct nvgpu_fd_translation_entry fd_translations[16];
  u32 num_fd_translations;

  /* ── DRI device nodes ── */
#define NVGPU_MAX_DRI_DEVS 8
  struct nvgpu_dri_dev dri_devs[NVGPU_MAX_DRI_DEVS];
  int num_dri_devs;

  /* ── PCI sysfs fake hierarchy ── */
  struct kobject *pci_bus_kobj;     /* /sys/bus/pci              */
  struct kobject *pci_devices_kobj; /* /sys/bus/pci/devices      */

#define NVGPU_MAX_PCI_SLOTS 8
  struct nvgpu_pci_root pci_roots[NVGPU_MAX_PCI_SLOTS];
  int num_pci_roots;
};

/*
 * Per-open-fd state.
 * Every open("/dev/nvidia*") creates one nvgpu_fd.
 * The VMM keeps a matching host FD identified by handle.
 */
struct nvgpu_fd {
  struct nvgpu_device *dev;
  u32 handle;      /* VMM-assigned handle from OPEN response */
  u32 device_type; /* NVGPU_DEV_*                            */
};

/* class for device_create() */
static struct class *nvgpu_class;

/* ───────── nvidia-drm stub — no DRM subsystem headers needed ───────── */

static long nvgpu_ioctl(struct file *filp, unsigned int cmd, unsigned long arg);

/*
 * We handle DRM ioctls without involving drm_ioctl() because that function
 * immediately casts filp->private_data to struct drm_file * and dereferences
 * ->minor->dev — our private_data is struct nvgpu_fd *, not struct drm_file *.
 *
 * All types used below are stable UAPI structs; we define only what we use.
 */

/* _IOC_TYPE byte for DRM ioctls */
#define DRM_IOCTL_BASE 'd'
#define DRM_COMMAND_BASE 0x40

/* DRM_NVIDIA_* are offsets from DRM_COMMAND_BASE */
#define DRM_NVIDIA_GET_DEV_INFO 0x03     /* abs nr 0x43 */
#define DRM_NVIDIA_FENCE_SUPPORTED 0x04  /* abs nr 0x44 */
#define DRM_NVIDIA_DMABUF_SUPPORTED 0x0f /* abs nr 0x4f */

/*
 * struct drm_version — UAPI, stable since DRM was upstreamed.
 * Copy of include/uapi/drm/drm.h:struct drm_version so we need
 * no DRM kernel headers.
 */
struct nvgpu_drm_version {
  int version_major;
  int version_minor;
  int version_patchlevel;
  size_t name_len;
  char __user *name;
  size_t date_len;
  char __user *date;
  size_t desc_len;
  char __user *desc;
};

struct drm_nvidia_get_dev_info_params {
  __u32 gpu_id;
  __u32 mig_device;
  __u32 primary_index;
  __u32 supports_alloc;
  __u32 generic_page_kind;
  __u32 page_kind_generation;
  __u32 sector_layout;
  __u32 supports_sync_fd;
  __u32 supports_semsurf;
} __packed;

/*
 * nvgpu_drm_handle_ioctl — handle all DRM-layer ioctls on our /dev/dri/..
 * nodes.
 *
 * DRM_IOCTL_VERSION  (nr=0x00) — core ioctl, returns name="nvidia-drm"
 * GET_DEV_INFO       (nr=0x43) — driver ioctl, returns gpu_id etc.
 * FENCE_SUPPORTED    (nr=0x44) — driver ioctl, returns 0
 * DMABUF_SUPPORTED   (nr=0x4f) — driver ioctl, returns 0
 *
 * Everything else → -ENOTTY.
 */
static long nvgpu_drm_handle_ioctl(struct nvgpu_fd *nfd,
                                   struct nvgpu_dri_dev *dri, unsigned int cmd,
                                   unsigned long arg) {
  unsigned int nr = _IOC_NR(cmd);
  void __user *uarg = (void __user *)arg;

  /* ── DRM_IOCTL_VERSION (type='d', nr=0x00) ── */
  if (nr == 0x00) {
    struct nvgpu_drm_version v;

    if (copy_from_user(&v, uarg, sizeof(v)))
      return -EFAULT;

    v.version_major = 0;
    v.version_minor = 1;
    v.version_patchlevel = 0;

#define FILL_DRM_STR(field, str)                                               \
  do {                                                                         \
    const char *_s = (str);                                                    \
    size_t _sl = strlen(_s);                                                   \
    if (v.field##_len >= _sl && v.field)                                       \
      if (copy_to_user(v.field, _s, _sl))                                      \
        return -EFAULT;                                                        \
    v.field##_len = _sl;                                                       \
  } while (0)

    FILL_DRM_STR(name, "nvidia-drm");
    FILL_DRM_STR(date, "20240101");
    FILL_DRM_STR(desc, "NVIDIA DRM stub");
#undef FILL_DRM_STR

    if (copy_to_user(uarg, &v, sizeof(v)))
      return -EFAULT;

    return 0;
  }

  /* ── Driver ioctls: DRM_COMMAND_BASE .. DRM_COMMAND_END ── */
  if (nr < DRM_COMMAND_BASE || nr >= DRM_COMMAND_END)
    return -ENOTTY;

  switch (nr - DRM_COMMAND_BASE) {

  case DRM_NVIDIA_GET_DEV_INFO: {
    struct drm_nvidia_get_dev_info_params p;

    memset(&p, 0, sizeof(p));
    p.gpu_id = dri->gpu_id;
    p.mig_device = 0;
    p.primary_index = 0;
    p.supports_alloc = 1;
    p.generic_page_kind = 6; /* Turing/Ampere */
    p.page_kind_generation = 2;
    p.sector_layout = 1;
    p.supports_sync_fd = 1;
    p.supports_semsurf = 1;

    if (copy_to_user(uarg, &p, sizeof(p)))
      return -EFAULT;
    return 0;
  }

  case DRM_NVIDIA_FENCE_SUPPORTED:
    return 0; /* not supported, no payload */

  case DRM_NVIDIA_DMABUF_SUPPORTED:
    return 0; /* not supported, no payload */

  default:
    return -ENOTTY;
  }
}

/*
 * nvgpu_dri_ioctl — combined ioctl for /dev/dri/.. nodes.
 *
 *   type 'd' → handled locally (DRM VERSION + nvidia-drm driver ioctls)
 *   type 'F' → proxied to host via virtqueue (NVIDIA RM ioctls)
 */
static long nvgpu_dri_ioctl(struct file *filp, unsigned int cmd,
                            unsigned long arg) {
  if (_IOC_TYPE(cmd) == DRM_IOCTL_BASE) {
    struct nvgpu_fd *nfd = filp->private_data;
    struct nvgpu_dri_dev *dri = NULL;
    int i;

    for (i = 0; i < nfd->dev->num_dri_devs; i++) {
      if ((u32)i == nfd->device_type - NVGPU_DEV_DRI_BASE) {
        dri = &nfd->dev->dri_devs[i];
        break;
      }
    }

    if (!dri)
      return -ENODEV;

    return nvgpu_drm_handle_ioctl(nfd, dri, cmd, arg);
  }

  /* NVIDIA-type and everything else → proxy to host */
  return nvgpu_ioctl(filp, cmd, arg);
}

/* ───────── Virtqueue communication ───────── */

/*
 * nvgpu_send_recv — submit one request to controlq and block until the VMM
 * returns the response.  Caller must supply pre-allocated resp buffer.
 */
static int nvgpu_send_recv(struct nvgpu_device *dev, void *req, int req_len,
                           void *resp, int resp_len) {
  struct scatterlist sg_out, sg_in;
  struct scatterlist *sgs[2] = {&sg_out, &sg_in};
  int ret;

  mutex_lock(&dev->vq_lock);

  reinit_completion(&dev->req_done);
  dev->resp_buf = resp;
  dev->resp_len = resp_len;

  sg_init_one(&sg_out, req, req_len);
  sg_init_one(&sg_in, resp, resp_len);

  ret = virtqueue_add_sgs(dev->ctrl_vq, sgs, 1, 1, resp, GFP_KERNEL);
  if (ret < 0) {
    mutex_unlock(&dev->vq_lock);
    return ret;
  }

  mutex_unlock(&dev->vq_lock);
  virtqueue_kick(dev->ctrl_vq);

  ret = wait_for_completion_killable_timeout(&dev->req_done, 10 * HZ);
  if (ret == 0) {
    return -ETIMEDOUT;
  }
  if (ret < 0) {
    return ret;
  }

  return 0;
}

/* Virtqueue callback: VMM has written the response buffer */
static void nvgpu_ctrl_vq_cb(struct virtqueue *vq) {
  struct nvgpu_device *dev = vq->vdev->priv;
  void *buf;
  unsigned int len;

  while ((buf = virtqueue_get_buf(vq, &len)) != NULL) {
    complete(&dev->req_done);
  }
}

/* event virtqueue callback — not used yet, just drain */
static void nvgpu_event_vq_cb(struct virtqueue *vq) {
  void *buf;
  unsigned int len;

  while ((buf = virtqueue_get_buf(vq, &len)) != NULL)
    /* TODO: deliver to waiting guest processes */;
}

/* ───────── Ioctl forwarding ───────── */

/* nvgpu_ioctl_simple — flat struct, no embedded pointers */
static long nvgpu_ioctl_simple(struct nvgpu_fd *nfd, unsigned int cmd,
                               void __user *uarg, unsigned int sz) {
  int req_total = sizeof(struct nvgpu_ioctl_req) + sz;
  int resp_max = sizeof(struct nvgpu_ioctl_resp) + sz;
  void *req_buf, *resp_buf;
  struct nvgpu_ioctl_req *req;
  struct nvgpu_ioctl_resp *resp;
  int ret;

  req_buf = kmalloc(req_total, GFP_KERNEL);
  resp_buf = kmalloc(resp_max, GFP_KERNEL);
  if (!req_buf || !resp_buf) {
    ret = -ENOMEM;
    goto out;
  }

  req = (struct nvgpu_ioctl_req *)req_buf;
  req->hdr.msg_type = cpu_to_le32(NVGPU_MSG_IOCTL);
  req->hdr.handle = cpu_to_le32(nfd->handle);
  req->hdr.status = 0;
  req->hdr.padding = 0;
  req->cmd = cpu_to_le32(cmd);
  req->data_len = cpu_to_le32(sz);
  req->nested_offset = 0;
  req->nested_len = 0;

  if (sz > 0) {
    if (copy_from_user(req_buf + sizeof(*req), uarg, sz)) {
      ret = -EFAULT;
      goto out;
    }
  }

  ret = nvgpu_send_recv(nfd->dev, req_buf, req_total, resp_buf, resp_max);
  if (ret < 0)
    goto out;

  resp = (struct nvgpu_ioctl_resp *)resp_buf;
  ret = (int)(s32)le32_to_cpu((__le32)resp->hdr.status);

  if (sz > 0 && resp->data_len && le32_to_cpu(resp->data_len) <= sz) {
    if (copy_to_user(uarg, resp_buf + sizeof(*resp),
                     le32_to_cpu(resp->data_len)))
      ret = -EFAULT;
  }

out:
  kfree(req_buf);
  kfree(resp_buf);
  return ret;
}

/*
 * nvgpu_ioctl_rm_control — NV_ESC_RM_CONTROL with nested params buffer.
 *
 * Handles three cases:
 *   1. Normal: nested params are flat data → marshal and forward
 *   2. V1→V2 rewrite: nested params contain a second userspace pointer
 *      that we can't forward → rewrite to V2 inline variant
 *   3. paramsSize == 0 or params == NULL → forward outer struct only
 */
static long nvgpu_ioctl_rm_control(struct nvgpu_fd *nfd, unsigned int cmd,
                                   void __user *uarg, unsigned int sz) {
  struct NVOS54_PARAMETERS params;
  void __user *user_nested;
  u32 nested_size;
  u32 ctl_cmd;
  const struct nvgpu_v1v2_entry *rw;

  /* V1→V2 rewrite state */
  u64 saved_user_ptr = 0;
  u32 saved_user_data_size = 0;
  u32 saved_v1_cmd = 0;
  u32 saved_v1_params_size = 0;

  void *req_buf = NULL, *resp_buf = NULL;
  struct nvgpu_ioctl_req *req;
  struct nvgpu_ioctl_resp *resp;
  int req_total, resp_max, ret;

  if (sz < sizeof(params))
    return -EINVAL;

  if (copy_from_user(&params, uarg, sizeof(params)))
    return -EFAULT;

  user_nested = (void __user *)(unsigned long)le64_to_cpu(params.params);
  nested_size = le32_to_cpu(params.paramsSize);
  ctl_cmd = le32_to_cpu(params.cmd);

  if (nested_size > 1024 * 1024)
    return -EINVAL;

  /* Intercept multi-pointer commands that can't be forwarded */
  {
    long intercept_ret;
    if (nvgpu_try_intercept_rm_control(nfd, ctl_cmd, uarg, user_nested,
                                       nested_size, nfd->dev->driver_version,
                                       &intercept_ret))
      return intercept_ret;
  }

  /*
   * Check for V1→V2 rewrite.  Only when we have nested params that
   * contain a second-level userspace pointer we can't forward.
   */
  rw = (user_nested && nested_size > 0) ? nvgpu_find_v1v2_rewrite(ctl_cmd)
                                        : NULL;

  if (rw) {
    void *v1_buf;
    u32 min_v1_size;

    /* Read V1 nested params from guest userspace */
    min_v1_size = rw->v1_userptr_offset + 8;
    if (nested_size < min_v1_size) {
      pr_warn(
          "virtio-gpu-nv: v1v2: V1 nested too small (%u < %u) for cmd 0x%x\n",
          nested_size, min_v1_size, ctl_cmd);
      rw = NULL;
      goto do_normal;
    }

    v1_buf = kmalloc(nested_size, GFP_KERNEL);
    if (!v1_buf)
      return -ENOMEM;

    if (copy_from_user(v1_buf, user_nested, nested_size)) {
      kfree(v1_buf);
      return -EFAULT;
    }

    /* Extract the userspace data pointer from V1 nested params */
    memcpy(&saved_user_ptr, v1_buf + rw->v1_userptr_offset, sizeof(u64));

    if (!saved_user_ptr) {
      pr_debug("virtio-gpu-nv: v1v2: data pointer is NULL for cmd 0x%x, normal "
               "path\n",
               ctl_cmd);
      kfree(v1_buf);
      rw = NULL;
      goto do_normal;
    }

    /* Compute how many bytes of result data guest expects back */
    if (rw->info_style && rw->v1_copy_prefix >= 4) {
      u32 list_size;
      memcpy(&list_size, v1_buf, sizeof(u32));
      saved_user_data_size = min_t(u32, list_size * 8, rw->v2_data_size);
    } else {
      u32 caps_tbl_size;
      memcpy(&caps_tbl_size, v1_buf, sizeof(u32));
      saved_user_data_size = min_t(u32, caps_tbl_size, rw->v2_data_size);
    }

    saved_v1_cmd = ctl_cmd;
    saved_v1_params_size = nested_size;

    pr_debug("virtio-gpu-nv: v1v2: rewriting cmd 0x%x → 0x%x (V2 %u bytes, "
             "data_back %u)\n",
             ctl_cmd, rw->v2_cmd, rw->v2_size, saved_user_data_size);

    /* Build V2 nested buffer (zeroed) */
    nested_size = rw->v2_size;

    req_total = sizeof(struct nvgpu_ioctl_req) + sizeof(params) + nested_size;
    resp_max = sizeof(struct nvgpu_ioctl_resp) + sizeof(params) + nested_size;

    req_buf = kzalloc(req_total, GFP_KERNEL);
    resp_buf = kzalloc(resp_max, GFP_KERNEL);
    if (!req_buf || !resp_buf) {
      kfree(v1_buf);
      ret = -ENOMEM;
      goto out;
    }

    /* Copy prefix from V1 into V2 (e.g. listSize or ceEngineType) */
    if (rw->v1_copy_prefix > 0) {
      u32 pfx = min_t(u32, rw->v1_copy_prefix, (u32)(nested_size));
      memcpy(req_buf + sizeof(struct nvgpu_ioctl_req) + sizeof(params), v1_buf,
             pfx);
    }

    kfree(v1_buf);

    /* Patch outer: replace cmd with V2, update paramsSize */
    params.cmd = cpu_to_le32(rw->v2_cmd);
    params.paramsSize = cpu_to_le32(nested_size);

    /* Fill request header */
    req = (struct nvgpu_ioctl_req *)req_buf;
    req->hdr.msg_type = cpu_to_le32(NVGPU_MSG_IOCTL);
    req->hdr.handle = cpu_to_le32(nfd->handle);
    req->hdr.status = 0;
    req->hdr.padding = 0;
    req->cmd = cpu_to_le32(cmd);
    req->data_len = cpu_to_le32(sizeof(params));
    req->nested_offset = cpu_to_le32(sizeof(params));
    req->nested_len = cpu_to_le32(nested_size);

    memcpy(req_buf + sizeof(*req), &params, sizeof(params));

    ret = nvgpu_send_recv(nfd->dev, req_buf, req_total, resp_buf, resp_max);
    if (ret < 0)
      goto out;

    resp = (struct nvgpu_ioctl_resp *)resp_buf;
    ret = (int)(s32)le32_to_cpu((__le32)resp->hdr.status);

    /* ── V1→V2 response path ── */

    /* Copy V2 inline result data back to guest's original userspace pointer */
    if (le32_to_cpu(resp->nested_len) > 0 && saved_user_ptr &&
        saved_user_data_size > 0) {
      u32 resp_nested = le32_to_cpu(resp->nested_len);
      u32 avail = 0;

      if (resp_nested > rw->v2_data_offset)
        avail = resp_nested - rw->v2_data_offset;

      if (avail > 0) {
        u32 copy_back = min_t(u32, saved_user_data_size, avail);
        if (copy_to_user((void __user *)saved_user_ptr,
                         resp_buf + sizeof(*resp) + sizeof(params) +
                             rw->v2_data_offset,
                         copy_back)) {
          ret = -EFAULT;
          goto out;
        }
      }
    }

    /* Restore V1 outer fields before copying back to userspace:
     * cmd → original V1 cmd, paramsSize → original, params ptr → original */
    {
      struct NVOS54_PARAMETERS *resp_params =
          (struct NVOS54_PARAMETERS *)(resp_buf + sizeof(*resp));
      resp_params->cmd = cpu_to_le32(saved_v1_cmd);
      resp_params->paramsSize = cpu_to_le32(saved_v1_params_size);
      resp_params->params = cpu_to_le64((u64)(unsigned long)user_nested);
    }

    if (copy_to_user(uarg, resp_buf + sizeof(*resp), sizeof(params))) {
      ret = -EFAULT;
      goto out;
    }

    goto out;
  }

do_normal:
  /* ── Normal path (no V1→V2 rewrite) ── */

  req_total = sizeof(struct nvgpu_ioctl_req) + sizeof(params) + nested_size;
  resp_max = sizeof(struct nvgpu_ioctl_resp) + sizeof(params) + nested_size;

  req_buf = kmalloc(req_total, GFP_KERNEL);
  resp_buf = kmalloc(resp_max, GFP_KERNEL);
  if (!req_buf || !resp_buf) {
    ret = -ENOMEM;
    goto out;
  }

  req = (struct nvgpu_ioctl_req *)req_buf;
  req->hdr.msg_type = cpu_to_le32(NVGPU_MSG_IOCTL);
  req->hdr.handle = cpu_to_le32(nfd->handle);
  req->hdr.status = 0;
  req->hdr.padding = 0;
  req->cmd = cpu_to_le32(cmd);
  req->data_len = cpu_to_le32(sizeof(params));
  req->nested_offset = cpu_to_le32(sizeof(params));
  req->nested_len = cpu_to_le32(nested_size);

  memcpy(req_buf + sizeof(*req), &params, sizeof(params));

  if (user_nested && nested_size > 0) {
    if (copy_from_user(req_buf + sizeof(*req) + sizeof(params), user_nested,
                       nested_size)) {
      ret = -EFAULT;
      goto out;
    }
  }

  ret = nvgpu_send_recv(nfd->dev, req_buf, req_total, resp_buf, resp_max);
  if (ret < 0)
    goto out;

  resp = (struct nvgpu_ioctl_resp *)resp_buf;
  ret = (int)(s32)le32_to_cpu((__le32)resp->hdr.status);

  if (copy_to_user(uarg, resp_buf + sizeof(*resp), sizeof(params))) {
    ret = -EFAULT;
    goto out;
  }

  if (user_nested && le32_to_cpu(resp->nested_len) > 0) {
    u32 copy_back = min(nested_size, le32_to_cpu(resp->nested_len));
    if (copy_to_user(user_nested, resp_buf + sizeof(*resp) + sizeof(params),
                     copy_back))
      ret = -EFAULT;
  }

out:
  kfree(req_buf);
  kfree(resp_buf);
  return ret;
}

/*
 * nvgpu_ioctl_rm_alloc — NV_ESC_RM_ALLOC, same pattern via NVOS64_PARAMETERS.
 *
 * Subtlety: when paramsSize == 0 but pAllocParms != NULL, the host RM
 * driver determines size from hClass.  We must look up the size ourselves
 * so we know how many bytes to copy_from_user.
 */
static long nvgpu_ioctl_rm_alloc(struct nvgpu_fd *nfd, unsigned int cmd,
                                 void __user *uarg, unsigned int sz) {
  struct NVOS64_PARAMETERS params;
  void __user *user_alloc;
  u32 nested_size;
  void *req_buf = NULL, *resp_buf = NULL, *nested;
  struct nvgpu_ioctl_req *req;
  struct nvgpu_ioctl_resp *resp;
  int req_total, resp_max, ret;

  if (sz < sizeof(params))
    return -EINVAL;

  if (copy_from_user(&params, uarg, sizeof(params)))
    return -EFAULT;

  user_alloc = (void __user *)(unsigned long)le64_to_cpu(params.pAllocParms);
  nested_size = le32_to_cpu(params.paramsSize);

  /*
   * When paramsSize == 0 but pAllocParms is non-NULL,
   * the host RM driver knows the size from hClass.  We need to copy
   * that many bytes from guest userspace so the VMM can forward them.
   */
  if (user_alloc && nested_size == 0) {
    u32 hClass = le32_to_cpu(params.hClass);
    nested_size = nvgpu_rmalloc_class_param_size(hClass);
    pr_debug(
        "virtio-gpu-nv: RM_ALLOC hClass=0x%04x paramsSize=0 → copy %u bytes\n",
        hClass, nested_size);
  }

  if (nested_size > 1024 * 1024)
    return -EINVAL;

  req_total = sizeof(*req) + sizeof(params) + nested_size;
  resp_max = sizeof(struct nvgpu_ioctl_resp) + sizeof(params) + nested_size;

  req_buf = kmalloc(req_total, GFP_KERNEL);
  resp_buf = kmalloc(resp_max, GFP_KERNEL);
  if (!req_buf || !resp_buf) {
    ret = -ENOMEM;
    goto out;
  }

  req = (struct nvgpu_ioctl_req *)req_buf;
  req->hdr.msg_type = cpu_to_le32(NVGPU_MSG_IOCTL);
  req->hdr.handle = cpu_to_le32(nfd->handle);
  req->hdr.status = 0;
  req->hdr.padding = 0;
  req->cmd = cpu_to_le32(cmd);
  req->data_len = cpu_to_le32(sizeof(params));
  req->nested_offset = cpu_to_le32(sizeof(params));
  req->nested_len = cpu_to_le32(nested_size);

  memcpy(req_buf + sizeof(*req), &params, sizeof(params));

  if (user_alloc && nested_size > 0) {
    nested = req_buf + sizeof(*req) + sizeof(params);
    if (copy_from_user(nested, user_alloc, nested_size)) {
      ret = -EFAULT;
      goto out;
    }
  }

  ret = nvgpu_send_recv(nfd->dev, req_buf, req_total, resp_buf, resp_max);
  if (ret < 0)
    goto out;

  resp = (struct nvgpu_ioctl_resp *)resp_buf;
  ret = (int)(s32)le32_to_cpu((__le32)resp->hdr.status);

  if (copy_to_user(uarg, resp_buf + sizeof(*resp), sizeof(params))) {
    ret = -EFAULT;
    goto out;
  }

  if (user_alloc && le32_to_cpu(resp->nested_len) > 0) {
    u32 copy_back = min(nested_size, le32_to_cpu(resp->nested_len));
    if (copy_to_user(user_alloc, resp_buf + sizeof(*resp) + sizeof(params),
                     copy_back))
      ret = -EFAULT;
  }

out:
  kfree(req_buf);
  kfree(resp_buf);
  return ret;
}

/*
 * NOTE: The only subtlety worth noting: copy_to_user on the way back writes the
 * VMM handle value (not a host fd number) back into the guest's buffer. That's
 * fine — nvidia-smi doesn't read the payload back after REGISTER_FD, it only
 * checks the return code. If a future fd-carrying ioctl does need the response
 * payload, the VMM would need to translate back from host fd → handle before
 * returning.
 */
static long nvgpu_ioctl_translate_fd(struct nvgpu_fd *nfd, unsigned int cmd,
                                     void __user *uarg, unsigned int sz,
                                     unsigned int payload_offset) {
  struct nvgpu_ioctl_req *req;
  struct nvgpu_ioctl_resp *resp;
  void *req_buf = NULL, *resp_buf = NULL;
  struct file *other_file;
  struct nvgpu_fd *other_nfd;
  int guest_fd;
  u32 host_handle;
  int req_total, resp_max, ret;

  if (sz < payload_offset + sizeof(guest_fd))
    return -EINVAL;

  req_total = sizeof(*req) + sz;
  resp_max = sizeof(*resp) + sz;

  req_buf = kmalloc(req_total, GFP_KERNEL);
  resp_buf = kmalloc(resp_max, GFP_KERNEL);
  if (!req_buf || !resp_buf) {
    ret = -ENOMEM;
    goto out;
  }

  /* Copy the full payload from userspace */
  if (copy_from_user(req_buf + sizeof(*req), uarg, sz)) {
    ret = -EFAULT;
    goto out;
  }

  /* Extract guest fd from its position in the payload */
  memcpy(&guest_fd, req_buf + sizeof(*req) + payload_offset, sizeof(guest_fd));

  /* Resolve guest fd → nvgpu_fd → VMM handle */
  other_file = fget(guest_fd);
  if (!other_file) {
    ret = -EBADF;
    goto out;
  }

  other_nfd = other_file->private_data;
  if (!other_nfd) {
    fput(other_file);
    ret = -EINVAL;
    goto out;
  }

  host_handle = other_nfd->handle;
  fput(other_file);

  /* Patch payload: replace raw guest fd with VMM handle */
  memcpy(req_buf + sizeof(*req) + payload_offset, &host_handle,
         sizeof(host_handle));

  /* Build request header */
  req = (struct nvgpu_ioctl_req *)req_buf;
  req->hdr.msg_type = cpu_to_le32(NVGPU_MSG_IOCTL);
  req->hdr.handle = cpu_to_le32(nfd->handle);
  req->hdr.status = 0;
  req->hdr.padding = 0;
  req->cmd = cpu_to_le32(cmd);
  req->data_len = cpu_to_le32(sz);
  req->nested_offset = 0;
  req->nested_len = 0;

  ret = nvgpu_send_recv(nfd->dev, req_buf, req_total, resp_buf, resp_max);
  if (ret < 0)
    goto out;

  resp = (struct nvgpu_ioctl_resp *)resp_buf;
  ret = (int)(s32)le32_to_cpu((__le32)resp->hdr.status);

  /* Write the (possibly modified) payload back to userspace */
  if (ret == 0 && le32_to_cpu(resp->data_len) > 0) {
    if (copy_to_user(uarg, resp_buf + sizeof(*resp),
                     le32_to_cpu(resp->data_len)))
      ret = -EFAULT;
  }

out:
  kfree(req_buf);
  kfree(resp_buf);
  return ret;
}

static const struct nvgpu_fd_translation_entry *
nvgpu_find_fd_translation(struct nvgpu_device *dev, unsigned int nr) {
  u32 i;
  for (i = 0; i < dev->num_fd_translations; i++)
    if (le32_to_cpu(dev->fd_translations[i].nr) == nr)
      return &dev->fd_translations[i];
  return NULL;
}

/* Main ioctl dispatcher */
static long nvgpu_ioctl(struct file *filp, unsigned int cmd,
                        unsigned long arg) {
  struct nvgpu_fd *nfd = filp->private_data;
  unsigned int nr = _IOC_NR(cmd);
  unsigned int sz = _IOC_SIZE(cmd);
  void __user *uarg = (void __user *)arg;
  const struct nvgpu_fd_translation_entry *fdt;

  /* Hard cap only — sz == 0 is valid for several NVIDIA ioctls
   * (e.g. NV_ESC_RM_FREE on some driver versions, and any ioctl
   * that encodes parameters via _IOC_NR only with no struct). */
  if (sz > 65536)
    return -EINVAL;

  fdt = nvgpu_find_fd_translation(nfd->dev, nr);
  if (fdt)
    return nvgpu_ioctl_translate_fd(nfd, cmd, uarg, sz,
                                    le32_to_cpu(fdt->payload_offset));

  switch (nr) {
  case NV_ESC_RM_CONTROL:
    return nvgpu_ioctl_rm_control(nfd, cmd, uarg, sz);
  case NV_ESC_RM_ALLOC:
    return nvgpu_ioctl_rm_alloc(nfd, cmd, uarg, sz);
  default:
    return nvgpu_ioctl_simple(nfd, cmd, uarg, sz);
  }
}

/* ───────── UVM ioctl ───────── */

static long nvgpu_uvm_ioctl(struct file *filp, unsigned int cmd,
                            unsigned long arg) {
  struct nvgpu_fd *nfd = filp->private_data;
  unsigned int nr = _IOC_NR(cmd);
  unsigned int sz = _IOC_SIZE(cmd);
  void __user *uarg = (void __user *)arg;

  /*
   * UVM ioctls use _IOC(0, 0, nr, 0x3000) — type=0, size=0x3000.
   * _IOC_SIZE() returns 0x3000 which is the max buffer, not the
   * actual struct size. Use the real struct sizes instead.
   *
   * UVM_INITIALIZE     (nr=1): flags:u64 + rmStatus:u32 + pad = 16 bytes
   * UVM_MM_INITIALIZE  (nr=2): uvmFd:s32 + rmStatus:u32        =  8 bytes
   *
   * For all other UVM ioctls we use 0x3000 as an upper bound since
   * we don't know their sizes — the host driver will only read what
   * it needs.
   */
  if (sz == 0 || sz == 0x3000) {
    switch (nr) {
    case 1:
      sz = 16;
      break; /* UVM_INITIALIZE        */
    case 2:
      sz = 8;
      break; /* UVM_MM_INITIALIZE     */
    default:
      sz = 0x3000;
      break;
    }
  }

  if (sz > 0x3000)
    return -EINVAL;

  /*
   * UVM_MM_INITIALIZE passes arg=0 (NULL) because the uvmFd is
   * embedded in the ioctl struct on some driver versions, or the
   * kernel side doesn't need userspace params at all.
   * Forward with a zeroed buffer — host will fill rmStatus.
   */
  if (arg == 0) {
    /*
     * Can't copy_from_user a NULL pointer. Build a zeroed buffer
     * and send it; the host UVM driver will populate rmStatus.
     */
    int req_total = sizeof(struct nvgpu_ioctl_req) + sz;
    int resp_max = sizeof(struct nvgpu_ioctl_resp) + sz;
    void *req_buf, *resp_buf;
    struct nvgpu_ioctl_req *req;
    struct nvgpu_ioctl_resp *resp;
    int ret;

    req_buf = kzalloc(req_total, GFP_KERNEL);
    resp_buf = kzalloc(resp_max, GFP_KERNEL);
    if (!req_buf || !resp_buf) {
      kfree(req_buf);
      kfree(resp_buf);
      return -ENOMEM;
    }

    req = (struct nvgpu_ioctl_req *)req_buf;
    req->hdr.msg_type = cpu_to_le32(NVGPU_MSG_IOCTL);
    req->hdr.handle = cpu_to_le32(nfd->handle);
    req->cmd = cpu_to_le32(cmd);
    req->data_len = cpu_to_le32(sz);
    /* payload stays zeroed — no copy_from_user */

    ret = nvgpu_send_recv(nfd->dev, req_buf, req_total, resp_buf, resp_max);

    if (ret == 0) {
      resp = (struct nvgpu_ioctl_resp *)resp_buf;
      ret = (int)(s32)le32_to_cpu((__le32)resp->hdr.status);
      /* arg=0 means no copy_to_user either */
    }

    kfree(req_buf);
    kfree(resp_buf);
    return ret;
  }

  /* UVM_INITIALIZE: inject MULTI_PROCESS_SHARING_MODE flag */
  if (nr == 1) {
    u64 flags;
    if (copy_from_user(&flags, uarg, sizeof(flags)))
      return -EFAULT;
    flags |= (1ULL << 2);
    if (copy_to_user(uarg, &flags, sizeof(flags)))
      return -EFAULT;
  }

  return nvgpu_ioctl_simple(nfd, cmd, uarg, sz);
}

/* ───────── mmap ───────── */

static void nvgpu_vma_close(struct vm_area_struct *vma) {
  struct nvgpu_fd *nfd = vma->vm_file->private_data;
  u32 mapping_id = (u32)(unsigned long)vma->vm_private_data;
  struct nvgpu_munmap_req *req;
  struct nvgpu_munmap_resp *resp;

  req = kzalloc(sizeof(*req), GFP_KERNEL);
  resp = kzalloc(sizeof(*resp), GFP_KERNEL);
  if (req && resp) {
    req->hdr.msg_type = cpu_to_le32(NVGPU_MSG_MUNMAP);
    req->hdr.handle = cpu_to_le32(nfd->handle);
    req->mapping_id = cpu_to_le32(mapping_id);
    nvgpu_send_recv(nfd->dev, req, sizeof(*req), resp, sizeof(*resp));
  }
  kfree(req);
  kfree(resp);
}

static const struct vm_operations_struct nvgpu_vm_ops = {
    .close = nvgpu_vma_close,
};

static int nvgpu_mmap(struct file *filp, struct vm_area_struct *vma) {
  struct nvgpu_fd *nfd = filp->private_data;
  u64 size = vma->vm_end - vma->vm_start;
  u64 offset = (u64)vma->vm_pgoff << PAGE_SHIFT;
  struct nvgpu_mmap_req *req;
  struct nvgpu_mmap_resp *resp;
  int ret;

  req = kzalloc(sizeof(*req), GFP_KERNEL);
  resp = kzalloc(sizeof(*resp), GFP_KERNEL);
  if (!req || !resp) {
    ret = -ENOMEM;
    goto out;
  }

  req->hdr.msg_type = cpu_to_le32(NVGPU_MSG_MMAP);
  req->hdr.handle = cpu_to_le32(nfd->handle);
  req->size = cpu_to_le64(size);
  req->offset = cpu_to_le64(offset);
  req->prot = cpu_to_le32((vma->vm_flags & VM_WRITE) ? 3 : 1);

  ret = nvgpu_send_recv(nfd->dev, req, sizeof(*req), resp, sizeof(*resp));
  if (ret < 0)
    goto out;

  if ((s32)le32_to_cpu((__le32)resp->hdr.status) < 0) {
    ret = (s32)le32_to_cpu((__le32)resp->hdr.status);
    goto out;
  }

  vm_flags_set(vma, VM_IO | VM_PFNMAP | VM_DONTEXPAND | VM_DONTDUMP);
  vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot);

  ret = remap_pfn_range(vma, vma->vm_start,
                        le64_to_cpu(resp->guest_phys_addr) >> PAGE_SHIFT, size,
                        vma->vm_page_prot);
  if (ret)
    goto out;

  vma->vm_ops = &nvgpu_vm_ops;
  vma->vm_private_data = (void *)(unsigned long)le32_to_cpu(resp->mapping_id);

out:
  kfree(req);
  kfree(resp);
  return ret;
}

/* ───────── open / release ───────── */

static int nvgpu_open_common(struct inode *inode, struct file *filp,
                             u32 device_type) {
  struct nvgpu_device *dev;
  struct nvgpu_fd *nfd;
  struct nvgpu_open_req *req;
  struct nvgpu_open_resp *resp;
  int ret;

  /* Recover nvgpu_device pointer depending on which cdev was opened */
  if (device_type == NVGPU_DEV_UVM || device_type == NVGPU_DEV_UVM_TOOLS)
    dev = container_of(inode->i_cdev, struct nvgpu_device, cdev_uvm);
  else if (device_type == NVGPU_DEV_CTL)
    dev = container_of(inode->i_cdev, struct nvgpu_device, cdev_ctl);
  else if (device_type == NVGPU_DEV_MODESET)
    dev = container_of(inode->i_cdev, struct nvgpu_device, cdev_modeset);
  else
    dev = container_of(inode->i_cdev, struct nvgpu_device,
                       cdev_gpu[iminor(inode)]);

  nfd = kzalloc(sizeof(*nfd), GFP_KERNEL);
  req = kzalloc(sizeof(*req), GFP_KERNEL);
  resp = kzalloc(sizeof(*resp), GFP_KERNEL);
  if (!nfd || !req || !resp) {
    kfree(nfd);
    kfree(req);
    kfree(resp);
    return -ENOMEM;
  }

  nfd->dev = dev;
  nfd->device_type = device_type;

  req->hdr.msg_type = cpu_to_le32(NVGPU_MSG_OPEN);
  req->device_type = cpu_to_le32(device_type);
  req->flags = cpu_to_le32(filp->f_flags);

  ret = nvgpu_send_recv(dev, req, sizeof(*req), resp, sizeof(*resp));
  if (ret < 0 || (s32)le32_to_cpu((__le32)resp->hdr.status) < 0) {
    kfree(nfd);
    kfree(req);
    kfree(resp);
    if (ret < 0)
      return ret;
    return (s32)le32_to_cpu((__le32)resp->hdr.status);
  }

  nfd->handle = le32_to_cpu(resp->hdr.handle);
  filp->private_data = nfd;
  kfree(req);
  kfree(resp);
  return 0;
}

static int nvgpu_gpu_open(struct inode *inode, struct file *filp) {
  return nvgpu_open_common(inode, filp, (u32)iminor(inode));
}

static int nvgpu_ctl_open(struct inode *inode, struct file *filp) {
  return nvgpu_open_common(inode, filp, NVGPU_DEV_CTL);
}

static int nvgpu_uvm_open(struct inode *inode, struct file *filp) {
  return nvgpu_open_common(inode, filp, NVGPU_DEV_UVM);
}

static int nvgpu_release(struct inode *inode, struct file *filp) {
  struct nvgpu_fd *nfd = filp->private_data;
  struct nvgpu_msg_hdr *req;
  struct nvgpu_msg_hdr *resp;

  req = kzalloc(sizeof(*req), GFP_KERNEL);
  resp = kzalloc(sizeof(*resp), GFP_KERNEL);
  if (req && resp) {
    req->msg_type = cpu_to_le32(NVGPU_MSG_CLOSE);
    req->handle = cpu_to_le32(nfd->handle);
    nvgpu_send_recv(nfd->dev, req, sizeof(*req), resp, sizeof(*resp));
  }
  kfree(req);
  kfree(resp);
  kfree(nfd);
  return 0;
}

/* ───────── file_operations tables ───────── */

static const struct file_operations nvgpu_gpu_fops = {
    .owner = THIS_MODULE,
    .open = nvgpu_gpu_open,
    .release = nvgpu_release,
    .unlocked_ioctl = nvgpu_ioctl,
    .mmap = nvgpu_mmap,
};

static const struct file_operations nvgpu_ctl_fops = {
    .owner = THIS_MODULE,
    .open = nvgpu_ctl_open,
    .release = nvgpu_release,
    .unlocked_ioctl = nvgpu_ioctl,
    .mmap = nvgpu_mmap,
};

static const struct file_operations nvgpu_uvm_fops = {
    .owner = THIS_MODULE,
    .open = nvgpu_uvm_open,
    .release = nvgpu_release,
    .unlocked_ioctl = nvgpu_uvm_ioctl,
    .mmap = nvgpu_mmap,
};

/* ───────── nvidia-modeset ioctl (/dev/nvidia-modeset, ioc_type 0x6d) ───────
 *
 * Outer struct (16 bytes):
 *   u32 cmd       — modeset sub-command
 *   u32 dataSize  — bytes pointed to by pData
 *   u64 pData     — USERSPACE pointer to the actual data buffer
 *
 * Same 2-level serialisation pattern as RM_CONTROL/RM_ALLOC.
 * The VMM side already handles pointer patching at offset 8 (see handler.rs).
 */

struct nvidia_modeset_outer {
  __le32 cmd;
  __le32 dataSize; /* ← the nested buffer size! */
  __le64 pData;    /* ← userspace pointer to nested params */
};

static long nvgpu_modeset_ioctl(struct file *filp, unsigned int cmd,
                                unsigned long arg) {
  struct nvgpu_fd *nfd = filp->private_data;
  void __user *uarg = (void __user *)arg;
  struct nvidia_modeset_outer outer;
  void __user *user_nested;
  u32 nested_size;
  void *req_buf = NULL, *resp_buf = NULL;
  struct nvgpu_ioctl_req *req;
  struct nvgpu_ioctl_resp *resp;
  int req_total, resp_max, ret;

  if (copy_from_user(&outer, uarg, sizeof(outer)))
    return -EFAULT;

  user_nested = (void __user *)(unsigned long)le64_to_cpu(outer.pData);
  nested_size = le32_to_cpu(outer.dataSize);

  if (nested_size > 1024 * 1024)
    return -EINVAL;

  req_total = sizeof(*req) + sizeof(outer) + nested_size;
  resp_max = sizeof(struct nvgpu_ioctl_resp) + sizeof(outer) + nested_size;

  req_buf = kmalloc(req_total, GFP_KERNEL);
  resp_buf = kmalloc(resp_max, GFP_KERNEL);
  if (!req_buf || !resp_buf) {
    ret = -ENOMEM;
    goto out;
  }

  req = (struct nvgpu_ioctl_req *)req_buf;
  req->hdr.msg_type = cpu_to_le32(NVGPU_MSG_IOCTL);
  req->hdr.handle = cpu_to_le32(nfd->handle);
  req->hdr.status = 0;
  req->hdr.padding = 0;
  req->cmd = cpu_to_le32(cmd);
  req->data_len = cpu_to_le32(sizeof(outer));
  req->nested_offset = cpu_to_le32(sizeof(outer));
  req->nested_len = cpu_to_le32(nested_size);

  memcpy(req_buf + sizeof(*req), &outer, sizeof(outer));

  if (user_nested && nested_size > 0) {
    if (copy_from_user(req_buf + sizeof(*req) + sizeof(outer), user_nested,
                       nested_size)) {
      ret = -EFAULT;
      goto out;
    }
  }

  ret = nvgpu_send_recv(nfd->dev, req_buf, req_total, resp_buf, resp_max);
  if (ret < 0)
    goto out;

  resp = (struct nvgpu_ioctl_resp *)resp_buf;
  ret = (int)(s32)le32_to_cpu((__le32)resp->hdr.status);

  /* Write back outer struct */
  if (copy_to_user(uarg, resp_buf + sizeof(*resp), sizeof(outer))) {
    ret = -EFAULT;
    goto out;
  }

  /* Write back nested params */
  if (user_nested && le32_to_cpu(resp->nested_len) > 0) {
    u32 copy_back = min(nested_size, le32_to_cpu(resp->nested_len));
    if (copy_to_user(user_nested, resp_buf + sizeof(*resp) + sizeof(outer),
                     copy_back))
      ret = -EFAULT;
  }

out:
  kfree(req_buf);
  kfree(resp_buf);
  return ret;
}

static long nvgpu_modeset_ioctl(struct file *filp, unsigned int cmd,
                                unsigned long arg) {
  struct nvgpu_fd *nfd = filp->private_data;
  unsigned int ioc_type = _IOC_TYPE(cmd);
  unsigned int sz = _IOC_SIZE(cmd);
  void __user *uarg = (void __user *)arg;

  if (sz > 65536)
    return -EINVAL;

  /* nvidia-modeset ioctls use type 0x6d ('m') */
  if (ioc_type == 0x6d)
    return nvgpu_ioctl_modeset(nfd, cmd, uarg, sz);

  /* Anything else (unlikely) falls back to the standard path */
  return nvgpu_ioctl(filp, cmd, arg);
}

static int nvgpu_modeset_open(struct inode *inode, struct file *filp) {
  return nvgpu_open_common(inode, filp, NVGPU_DEV_MODESET);
}

static const struct file_operations nvgpu_modeset_fops = {
    .owner = THIS_MODULE,
    .open = nvgpu_modeset_open,
    .release = nvgpu_release,
    .unlocked_ioctl = nvgpu_modeset_ioctl,
    .mmap = nvgpu_mmap,
};

/* ───────── /proc/driver/nvidia ───────── */

static int nvgpu_proc_version_show(struct seq_file *m, void *v) {
  struct nvgpu_device *dev = m->private;

  seq_printf(m,
             "NVRM version: NVIDIA UNIX x86_64 Kernel Module  %s\n"
             "GCC version:  gcc version 12.2.0\n",
             dev->driver_version);
  return 0;
}

static int nvgpu_proc_version_open(struct inode *inode, struct file *filp) {
  return single_open(filp, nvgpu_proc_version_show, pde_data(inode));
}

static const struct proc_ops nvgpu_proc_version_ops = {
    .proc_open = nvgpu_proc_version_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

static int nvgpu_proc_params_show(struct seq_file *m, void *v) {
  struct nvgpu_device *dev = m->private;

  seq_printf(m, "NVreg_EnablePCIeGen3=1\n"
                "NVreg_MemoryPoolSize=0\n");
  (void)dev;
  return 0;
}

static int nvgpu_proc_params_open(struct inode *inode, struct file *filp) {
  return single_open(filp, nvgpu_proc_params_show, pde_data(inode));
}

static const struct proc_ops nvgpu_proc_params_ops = {
    .proc_open = nvgpu_proc_params_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

/* Generic heap-backed proc file — used for all passthrough files */
struct nvgpu_proc_buf {
  char *data;
  size_t len;
};

static int nvgpu_proc_buf_show(struct seq_file *m, void *v) {
  struct nvgpu_proc_buf *b = m->private;
  seq_write(m, b->data, b->len);
  return 0;
}

static int nvgpu_proc_buf_open(struct inode *inode, struct file *filp) {
  return single_open(filp, nvgpu_proc_buf_show, pde_data(inode));
}

static const struct proc_ops nvgpu_proc_buf_ops = {
    .proc_open = nvgpu_proc_buf_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = single_release,
};

/* Simple directory cache — avoids duplicate proc_mkdir calls */
#define NVGPU_PROC_MAX_DIRS 32

struct nvgpu_proc_dir_cache {
  char path[128];
  struct proc_dir_entry *entry;
};

static struct nvgpu_proc_dir_cache nvgpu_dir_cache[NVGPU_PROC_MAX_DIRS];
static int nvgpu_dir_cache_count;

static void nvgpu_dir_cache_reset(void) {
  memset(nvgpu_dir_cache, 0, sizeof(nvgpu_dir_cache));
  nvgpu_dir_cache_count = 0;
}

static struct proc_dir_entry *
nvgpu_proc_mkdir_cached(const char *path, struct proc_dir_entry *parent) {
  int i;

  /* Check cache first */
  for (i = 0; i < nvgpu_dir_cache_count; i++) {
    if (strcmp(nvgpu_dir_cache[i].path, path) == 0)
      return nvgpu_dir_cache[i].entry;
  }

  /* Not cached — create it */
  struct proc_dir_entry *entry = proc_mkdir(path, parent);

  /* Cache it even if NULL — so we don't retry failed creates */
  if (nvgpu_dir_cache_count < NVGPU_PROC_MAX_DIRS) {
    strscpy(nvgpu_dir_cache[nvgpu_dir_cache_count].path, path, 128);
    nvgpu_dir_cache[nvgpu_dir_cache_count].entry = entry;
    nvgpu_dir_cache_count++;
  }

  return entry;
}

static struct proc_dir_entry *nvgpu_proc_mkdir_parents(char *pathbuf,
                                                       char **leaf_name) {
  struct proc_dir_entry *parent = NULL;
  char built[256] = {};
  char *slash;
  char *p;

  slash = strrchr(pathbuf, '/');
  if (!slash) {
    *leaf_name = pathbuf;
    return NULL;
  }

  *leaf_name = slash + 1;
  *slash = '\0';

  /* Walk each component, building the full path as we go
   * so the cache key is always the full absolute component */
  p = pathbuf;
  while (*p) {
    char *next = strchr(p, '/');
    if (next)
      *next = '\0';

    /* Append component to built path */
    if (built[0])
      strlcat(built, "/", sizeof(built));
    strlcat(built, p, sizeof(built));

    parent = nvgpu_proc_mkdir_cached(built, NULL);

    if (next) {
      *next = '/';
      p = next + 1;
    } else {
      break;
    }
  }

  return parent;
}

static int nvgpu_proc_init(struct nvgpu_device *dev) {
  struct nvgpu_msg_hdr *req;
  u8 *resp_buf, *p, *end;
  /* 512 KiB — vastly more than needed, avoids any size guessing */
  const size_t resp_size = 512 * 1024;
  int ret = 0;

  req = kzalloc(sizeof(*req), GFP_KERNEL);
  if (!req)
    return -ENOMEM;

  resp_buf = kvmalloc(resp_size, GFP_KERNEL);
  if (!resp_buf) {
    kfree(req);
    return -ENOMEM;
  }

  req->msg_type = cpu_to_le32(NVGPU_MSG_GET_PROC_FILES);
  req->handle = 0;
  req->status = 0;
  req->padding = 0;

  ret = nvgpu_send_recv(dev, req, sizeof(*req), resp_buf, resp_size);
  if (ret < 0) {
    dev_err(&dev->vdev->dev, "virtio-gpu-nv: GET_PROC_FILES failed: %d\n", ret);
    goto out;
  }

  p = resp_buf;
  end = resp_buf + resp_size;

  nvgpu_dir_cache_reset();

  while (p + 8 <= end) {
    u32 path_len, content_len;
    struct nvgpu_proc_buf *buf;
    char *pathbuf, *leaf;
    struct proc_dir_entry *parent = NULL;

    memcpy(&path_len, p, 4);
    path_len = le32_to_cpu((__le32)path_len);
    memcpy(&content_len, p + 4, 4);
    content_len = le32_to_cpu((__le32)content_len);
    p += 8;

    if (path_len == 0)
      break; /* terminator */

    if (p + path_len + content_len > end) {
      dev_warn(&dev->vdev->dev, "virtio-gpu-nv: proc stream truncated\n");
      break;
    }

    buf = kzalloc(sizeof(*buf), GFP_KERNEL);
    if (!buf) {
      ret = -ENOMEM;
      goto out;
    }

    buf->data = kmemdup(p + path_len, content_len, GFP_KERNEL);
    if (!buf->data) {
      kfree(buf);
      ret = -ENOMEM;
      goto out;
    }
    buf->len = content_len;

    pathbuf = kmalloc(path_len + 1, GFP_KERNEL);
    if (!pathbuf) {
      kfree(buf->data);
      kfree(buf);
      ret = -ENOMEM;
      goto out;
    }
    memcpy(pathbuf, p, path_len);
    pathbuf[path_len] = '\0';

    parent = nvgpu_proc_mkdir_parents(pathbuf, &leaf);
    proc_create_data(leaf, 0444, parent, &nvgpu_proc_buf_ops, buf);
    dev_dbg(&dev->vdev->dev, "virtio-gpu-nv: /proc/%s (%u bytes)\n", pathbuf,
            content_len);

    kfree(pathbuf);
    p += path_len + content_len;
  }

out:
  kvfree(resp_buf);
  kfree(req);
  return ret;
}

/* ───────── DRI device nodes (host major:minor passthrough) ─────────────── */

static int nvgpu_dri_open(struct inode *inode, struct file *filp) {
  struct nvgpu_dri_dev *dri =
      container_of(inode->i_cdev, struct nvgpu_dri_dev, cdev);
  struct nvgpu_device *dev = dri->dev;
  struct nvgpu_fd *nfd = NULL;
  struct nvgpu_open_req *req = NULL;
  struct nvgpu_open_resp *resp = NULL;
  int ret;

  if (!dev)
    return -ENODEV;

  nfd = kzalloc(sizeof(*nfd), GFP_KERNEL);
  req = kzalloc(sizeof(*req), GFP_KERNEL);
  resp = kzalloc(sizeof(*resp), GFP_KERNEL);
  if (!nfd || !req || !resp) {
    ret = -ENOMEM;
    goto err;
  }

  nfd->dev = dev;
  nfd->device_type = NVGPU_DEV_DRI_BASE + dri->index;

  req->hdr.msg_type = cpu_to_le32(NVGPU_MSG_OPEN);
  req->hdr.handle = 0;
  req->hdr.status = 0;
  req->hdr.padding = 0;
  req->device_type = cpu_to_le32(NVGPU_DEV_DRI_BASE + dri->index);
  req->flags = cpu_to_le32(filp->f_flags);

  ret = nvgpu_send_recv(dev, req, sizeof(*req), resp, sizeof(*resp));
  if (ret < 0)
    goto err;

  if ((s32)le32_to_cpu((__le32)resp->hdr.status) < 0) {
    ret = (s32)le32_to_cpu((__le32)resp->hdr.status);
    goto err;
  }

  nfd->handle = le32_to_cpu(resp->hdr.handle);
  filp->private_data = nfd;
  kfree(req);
  kfree(resp);
  return 0;

err:
  kfree(nfd);
  kfree(req);
  kfree(resp);
  return ret;
}

static const struct file_operations nvgpu_dri_fops = {
    .owner = THIS_MODULE,
    .open = nvgpu_dri_open,
    .release = nvgpu_release,
    .unlocked_ioctl = nvgpu_dri_ioctl,
    .mmap = nvgpu_mmap,
};

static char *nvgpu_devnode(const struct device *dev, umode_t *mode) {
  if (mode)
    *mode = 0666;
  return NULL;
}

static struct class *nvgpu_dri_class;

static char *nvgpu_dri_devnode(const struct device *dev, umode_t *mode) {
  if (mode)
    *mode = 0666;
  return kasprintf(GFP_KERNEL, "dri/%s", dev_name(dev));
}

static int nvgpu_dri_init(struct nvgpu_device *dev) {
  int i;

  if (dev->num_dri_devs == 0) {
    dev_info(&dev->vdev->dev,
             "virtio-gpu-nv: no DRI devices reported by VMM\n");
    return 0;
  }

  nvgpu_dri_class = class_create("nvgpu_dri");
  if (IS_ERR(nvgpu_dri_class)) {
    dev_err(&dev->vdev->dev, "virtio-gpu-nv: failed to create dri class: %ld\n",
            PTR_ERR(nvgpu_dri_class));
    nvgpu_dri_class = NULL;
    return PTR_ERR(nvgpu_dri_class);
  }
  nvgpu_dri_class->devnode = nvgpu_dri_devnode;

  for (i = 0; i < dev->num_dri_devs; i++) {
    struct nvgpu_dri_dev *dri = &dev->dri_devs[i];
    dev_t devno = MKDEV(dri->major, dri->minor);

    /*
     * Find the pci_dev that owns this DRI device so we can:
     *   a) Use it as the parent of the device_create() call — this causes
     *      the kernel to create /sys/dev/char/M:N/device → pci_dev, which
     *      is what Vulkan/EGL reads when it traverses the sysfs char-dev tree.
     *   b) Create drm/<name> kobjects under the PCI device, which gives
     *      /sys/bus/pci/devices/<addr>/drm/<name> — required by the NVIDIA
     *      Vulkan ICD when it enumerates display engines.
     *
     * We match by gpu_id (minor number) against the GPU slots in config space.
     */
    struct device *pci_parent = &dev->vdev->dev; /* fallback */
    struct kobject *pci_kobj = NULL;
    int gi;

    for (gi = 0; gi < dev->num_pci_roots; gi++) {
      struct nvgpu_pci_root *root = &dev->pci_roots[gi];

      if (!root->registered || !root->pdev)
        continue;

      /* Match: the DRI device belongs to this GPU if the GPU's minor number
       * (which equals the /dev/nvidia<minor> index) matches the gpu_id field
       * set from the host.  gpu_id is the 32-bit RM client GPU identifier,
       * but we stored minor there from the VMM side — see device.rs. */
      {
        u32 slot_minor = le32_to_cpu(dev->gpu_slots[gi].minor);
        if (slot_minor != dri->gpu_id && gi != 0)
          continue; /* only fall through for GPU 0 as a last resort */
      }

      pci_parent = &root->pdev->dev;
      pci_kobj = &root->pdev->dev.kobj;
      break;
    }

    /*
     * Create /sys/devices/pci.../0000:08:00.0/drm/<name> so the Vulkan ICD
     * can traverse: /sys/dev/char/226:N/device/drm/<name>
     *
     * We create a "drm" kobject under the PCI device's kobject (if we found
     * the pdev), then a child kobject named after the DRI node (renderD128,
     * card1, etc.).  These are bare kobjects — no attributes needed; just the
     * directory entries are enough for stat() calls from Vulkan.
     */
    if (pci_kobj) {
      /* "drm" dir — reuse if multiple DRI nodes share the same GPU */
      struct kobject *drm_parent = NULL;
      int j;

      for (j = 0; j < i; j++) {
        if (dev->dri_devs[j].drm_kobj &&
            dev->dri_devs[j].drm_kobj->parent == pci_kobj) {
          drm_parent = dev->dri_devs[j].drm_kobj;
          break;
        }
      }

      if (!drm_parent) {
        drm_parent = kobject_create_and_add("drm", pci_kobj);
        if (!drm_parent)
          dev_warn(&dev->vdev->dev,
                   "virtio-gpu-nv: failed to create drm kobj for %s\n",
                   dri->name);
      }

      dri->drm_kobj = drm_parent;

      if (drm_parent) {
        dri->drm_node_kobj = kobject_create_and_add(dri->name, drm_parent);
        if (!dri->drm_node_kobj)
          dev_warn(&dev->vdev->dev,
                   "virtio-gpu-nv: failed to create drm/%s kobj\n", dri->name);
        else
          dev_info(&dev->vdev->dev,
                   "virtio-gpu-nv: created sysfs drm/%s under PCI device\n",
                   dri->name);
      }
    }

    dri->index = (u32)i;
    dri->dev = dev;

    /* ── Register the cdev at host major:minor ── */
    if (register_chrdev_region(devno, 1, dri->name) != 0) {
      dev_warn(&dev->vdev->dev, "virtio-gpu-nv: cannot reserve %s (%u:%u)\n",
               dri->name, dri->major, dri->minor);
      continue;
    }

    cdev_init(&dri->cdev, &nvgpu_dri_fops);
    dri->cdev.owner = THIS_MODULE;

    if (cdev_add(&dri->cdev, devno, 1) != 0) {
      unregister_chrdev_region(devno, 1);
      dev_warn(&dev->vdev->dev, "virtio-gpu-nv: cdev_add %s failed\n",
               dri->name);
      continue;
    }

    /*
     * Use the PCI device as the parent.  This causes udev to create:
     *   /sys/class/nvgpu_dri/<name>/device  → <pci_dev sysfs path>
     * and the kernel to create:
     *   /sys/dev/char/M:N                   → ../../class/nvgpu_dri/<name>
     *
     * The Vulkan ICD then resolves:
     *   /sys/dev/char/M:N/device/drm/<name> → the kobject we made above.
     */
    device_create(nvgpu_dri_class, pci_parent, devno, NULL, "%s", dri->name);

    dri->registered = true;
    dev_info(&dev->vdev->dev,
             "virtio-gpu-nv: registered /dev/dri/%s (%u:%u) gpu_id=0x%x\n",
             dri->name, dri->major, dri->minor, dri->gpu_id);
  }

  return 0;
}

static void nvgpu_dri_cleanup(struct nvgpu_device *dev) {
  int i;

  for (i = 0; i < dev->num_dri_devs; i++) {
    struct nvgpu_dri_dev *dri = &dev->dri_devs[i];

    /* Tear down sysfs drm kobjects first */
    if (dri->drm_node_kobj) {
      kobject_put(dri->drm_node_kobj);
      dri->drm_node_kobj = NULL;
    }
    /* Only put the shared drm_kobj from the first node that created it */
    if (dri->drm_kobj) {
      int j;
      bool is_owner = true;
      for (j = 0; j < i; j++) {
        if (dev->dri_devs[j].drm_kobj == dri->drm_kobj) {
          is_owner = false;
          break;
        }
      }
      if (is_owner)
        kobject_put(dri->drm_kobj);
      dri->drm_kobj = NULL;
    }

    if (!dri->registered)
      continue;

    device_destroy(nvgpu_dri_class, MKDEV(dri->major, dri->minor));
    cdev_del(&dri->cdev);
    unregister_chrdev_region(MKDEV(dri->major, dri->minor), 1);
    dri->registered = false;
  }

  if (nvgpu_dri_class) {
    class_destroy(nvgpu_dri_class);
    nvgpu_dri_class = NULL;
  }
}

/* --- SYSTEM BUS PCI DEVS --- */

static int nvgpu_pci_read(struct pci_bus *bus, unsigned int devfn, int where,
                          int size, u32 *val) {
  struct nvgpu_pci_root *root = bus->sysdata;
  u8 slot = PCI_SLOT(devfn);
  u8 func = PCI_FUNC(devfn);

  /* Only respond to our specific device */
  if (slot != root->slot.slot || func != root->slot.func) {
    *val = ~0u;
    return PCIBIOS_DEVICE_NOT_FOUND;
  }

  if (!root->slot.config_valid || where + size > 256) {
    *val = ~0u;
    return PCIBIOS_BAD_REGISTER_NUMBER;
  }

  switch (size) {
  case 1:
    *val = root->slot.config[where];
    break;
  case 2:
    *val = le16_to_cpu(*(u16 *)&root->slot.config[where]);
    break;
  case 4:
    *val = le32_to_cpu(*(u32 *)&root->slot.config[where]);
    break;
  default:
    *val = ~0u;
    return PCIBIOS_BAD_REGISTER_NUMBER;
  }
  return PCIBIOS_SUCCESSFUL;
}

static int nvgpu_pci_write(struct pci_bus *bus, unsigned int devfn, int where,
                           int size, u32 val) {
  /* Config space is read-only from guest perspective */
  return PCIBIOS_FUNC_NOT_SUPPORTED;
}

static struct pci_ops nvgpu_pci_ops = {
    .read = nvgpu_pci_read,
    .write = nvgpu_pci_write,
};

/* Parse "DDDD:BB:SS.F" into components.
 * Returns 0 on success. */
static int nvgpu_parse_pci_addr(const char *addr, u16 *domain, u8 *bus,
                                u8 *slot, u8 *func) {
  unsigned int d, b, s, f;

  if (sscanf(addr, "%04x:%02x:%02x.%1x", &d, &b, &s, &f) != 4)
    return -EINVAL;

  *domain = (u16)d;
  *bus = (u8)b;
  *slot = (u8)s;
  *func = (u8)f;
  return 0;
}

static int nvgpu_pci_init(struct nvgpu_device *dev) {
  int i, ret = 0;

  for (i = 0; i < dev->num_pci_roots; i++) {
    struct nvgpu_pci_root *root = &dev->pci_roots[i];
    struct pci_host_bridge *bridge;
    struct resource *bus_res;

    if (!root->slot.config_valid) {
      dev_warn(&dev->vdev->dev,
               "virtio-gpu-nv: no config space for %s, skipping\n",
               root->slot.pci_addr);
      continue;
    }

    bridge = pci_alloc_host_bridge(0);
    if (!bridge) {
      dev_err(&dev->vdev->dev,
              "virtio-gpu-nv: pci_alloc_host_bridge failed for %s\n",
              root->slot.pci_addr);
      ret = -ENOMEM;
      continue;
    }

    /* One bus resource covering exactly our bus number */
    bus_res = kzalloc(sizeof(*bus_res), GFP_KERNEL);
    if (!bus_res) {
      pci_free_host_bridge(bridge);
      ret = -ENOMEM;
      continue;
    }
    bus_res->start = root->slot.bus_nr;
    bus_res->end = root->slot.bus_nr;
    bus_res->flags = IORESOURCE_BUS;
    pci_add_resource(&bridge->windows, bus_res);

    bridge->dev.parent = &dev->vdev->dev;
    root->domain = (int)root->slot.domain;
    bridge->sysdata = root;
    bridge->ops = &nvgpu_pci_ops;
    bridge->busnr = root->slot.bus_nr;
    bridge->domain_nr = root->slot.domain; /* parsed u16, not ASCII bytes */
    root->nvdev = dev;

    ret = pci_scan_root_bus_bridge(bridge);
    if (ret) {
      dev_err(&dev->vdev->dev,
              "virtio-gpu-nv: pci_scan_root_bus_bridge %s: %d\n",
              root->slot.pci_addr, ret);
      pci_free_host_bridge(bridge);
      kfree(bus_res);
      continue;
    }

    pci_bus_add_devices(bridge->bus);
    root->bridge = bridge;
    root->registered = true;

    /* Save the one pci_dev on this bus so DRI init can use it as a parent */
    {
      struct pci_dev *pdev;
      list_for_each_entry(pdev, &bridge->bus->devices, bus_list) {
        root->pdev = pdev;
        break;
      }
    }

    dev_info(&dev->vdev->dev, "virtio-gpu-nv: registered fake PCI device %s\n",
             root->slot.pci_addr);
  }

  return ret;
}

static void nvgpu_pci_cleanup(struct nvgpu_device *dev) {
  int i;

  for (i = 0; i < dev->num_pci_roots; i++) {
    struct nvgpu_pci_root *root = &dev->pci_roots[i];

    if (!root->registered)
      continue;

    pci_remove_root_bus(root->bridge->bus);
    /* pci_remove_root_bus frees the bridge */
    root->bridge = NULL;
    root->registered = false;
  }
}

/* ───────── GET_SYS_FILES handler (guest side) ──────────────────────────── */

static int nvgpu_fetch_sys_files(struct nvgpu_device *dev) {
  struct nvgpu_msg_hdr *req;
  u8 *resp_buf;
  u8 *p, *end;
  const int resp_max = 128 * 1024;
  int ret = 0;

  req = kzalloc(sizeof(*req), GFP_KERNEL);
  if (!req)
    return -ENOMEM;

  /* kvzalloc — zeroed so unwritten tail is never misread as data */
  resp_buf = kvzalloc(resp_max, GFP_KERNEL);
  if (!resp_buf) {
    kfree(req);
    return -ENOMEM;
  }

  req->msg_type = cpu_to_le32(NVGPU_MSG_GET_SYS_FILES);
  req->handle = 0;
  req->status = 0;
  req->padding = 0;

  ret = nvgpu_send_recv(dev, req, sizeof(*req), resp_buf, resp_max);
  if (ret < 0)
    goto out;

  p = resp_buf;
  end = resp_buf + resp_max;

  /* ── Section 1: sysfs files ─────────────────────────────────────── */
  while (p + 8 <= end) {
    /* Fix: memcpy for unaligned u32 reads, matching nvgpu_proc_init style */
    __le32 raw_path_len, raw_content_len;
    u32 path_len, content_len, copy_len;
    char path[256];

    memcpy(&raw_path_len, p, sizeof(__le32));
    memcpy(&raw_content_len, p + 4, sizeof(__le32));
    path_len = le32_to_cpu(raw_path_len);
    content_len = le32_to_cpu(raw_content_len);
    p += 8;

    if (path_len == 0 && content_len == 0)
      break; /* terminator */

    if (p + path_len + content_len > end) {
      dev_warn(&dev->vdev->dev,
               "virtio-gpu-nv: sys stream truncated at sysfs section\n");
      break;
    }

    /* Safe path extraction — explicit memset, no {} initialiser */
    memset(path, 0, sizeof(path));
    copy_len = min(path_len, (u32)(sizeof(path) - 1));
    memcpy(path, p, copy_len);
    p += path_len;

    if (p + content_len > end) {
      dev_warn(&dev->vdev->dev,
               "virtio-gpu-nv: sys stream truncated at content\n");
      break;
    }

    if (strncmp(path, "bus/pci/devices/", 16) == 0) {
      char *rest = path + 16; /* "<addr>/<filename>" */
      char *slash = strchr(rest, '/');

      if (slash && strcmp(slash + 1, "config") == 0) {
        char pci_addr[16] = {};
        int pi;

        memcpy(pci_addr, rest,
               min((size_t)(slash - rest), sizeof(pci_addr) - 1));

        /* Find existing slot or allocate new one */
        for (pi = 0; pi < dev->num_pci_roots; pi++)
          if (strcmp(dev->pci_roots[pi].slot.pci_addr, pci_addr) == 0)
            break;

        /* Match against known GPU slots to avoid creating
         * entries for unrelated PCI devices */
        if (pi == dev->num_pci_roots) {
          int gi;
          for (gi = 0; gi < (int)dev->num_gpus; gi++) {
            if (strcmp(dev->gpu_slots[gi].pci_addr, pci_addr) == 0) {
              pi = dev->num_pci_roots;
              if (pi < NVGPU_MAX_PCI_SLOTS) {
                memcpy(dev->pci_roots[pi].slot.pci_addr, pci_addr,
                       sizeof(pci_addr));
                if (nvgpu_parse_pci_addr(pci_addr,
                                         &dev->pci_roots[pi].slot.domain,
                                         &dev->pci_roots[pi].slot.bus_nr,
                                         &dev->pci_roots[pi].slot.slot,
                                         &dev->pci_roots[pi].slot.func) == 0)
                  dev->num_pci_roots++;
                else
                  pi = dev->num_pci_roots; /* parse failed */
              }
              break;
            }
          }
        }

        if (pi < dev->num_pci_roots) {
          struct nvgpu_pci_slot *ps = &dev->pci_roots[pi].slot;
          u32 copy = min(content_len, (u32)sizeof(ps->config));
          memcpy(ps->config, p, copy);
          ps->config_valid = true;
          dev_dbg(&dev->vdev->dev,
                  "virtio-gpu-nv: stored config space for %s (%u bytes)\n",
                  pci_addr, copy);
        }
      }
      /* Other PCI sysfs files (vendor, device, etc.) are handled
       * automatically by the kernel once the pci_dev is registered */
    }
    /* Unknown paths silently skipped */

    p += content_len;
  }

  /* ── Section 2: DRI devices ─────────────────────────────────────── */
  if (p + 4 > end) {
    dev_warn(&dev->vdev->dev,
             "virtio-gpu-nv: sys stream truncated before DRI section\n");
    goto out;
  }

  {
    __le32 raw_num_dri;
    u32 num_dri, i;

    memcpy(&raw_num_dri, p, sizeof(__le32));
    num_dri = le32_to_cpu(raw_num_dri);
    p += 4;

    num_dri = min(num_dri, (u32)NVGPU_MAX_DRI_DEVS);
    dev->num_dri_devs = 0;

    for (i = 0; i < num_dri; i++) {
      __le32 raw_name_len, raw_major, raw_minor, raw_gpu_id;
      u32 name_len, major, minor, gpu_id, nl;
      int idx;

      /* 16 bytes: name_len + major + minor + gpu_id */
      if (p + 16 > end) {
        dev_warn(&dev->vdev->dev,
                 "virtio-gpu-nv: DRI section truncated at entry %u\n", i);
        break;
      }

      memcpy(&raw_name_len, p, sizeof(__le32));
      memcpy(&raw_major, p + 4, sizeof(__le32));
      memcpy(&raw_minor, p + 8, sizeof(__le32));
      memcpy(&raw_gpu_id, p + 12, sizeof(__le32)); /* ← new */
      name_len = le32_to_cpu(raw_name_len);
      major = le32_to_cpu(raw_major);
      minor = le32_to_cpu(raw_minor);
      gpu_id = le32_to_cpu(raw_gpu_id); /* ← new */
      p += 16;                          /* was 12 */

      if (name_len == 0 || p + name_len > end) {
        dev_warn(&dev->vdev->dev,
                 "virtio-gpu-nv: DRI entry %u bad name_len %u\n", i, name_len);
        break;
      }

      idx = dev->num_dri_devs;
      nl = min(name_len, (u32)(sizeof(dev->dri_devs[idx].name) - 1));
      memset(dev->dri_devs[idx].name, 0, sizeof(dev->dri_devs[idx].name));
      memcpy(dev->dri_devs[idx].name, p, nl);
      dev->dri_devs[idx].major = major;
      dev->dri_devs[idx].minor = minor;
      dev->dri_devs[idx].gpu_id = gpu_id; /* ← new */
      dev->num_dri_devs++;

      dev_info(&dev->vdev->dev,
               "virtio-gpu-nv: DRI %s (%u:%u) gpu_id=0x%x from host\n",
               dev->dri_devs[idx].name, major, minor, gpu_id);
      p += name_len;
    }
  }

out:
  kvfree(resp_buf);
  kfree(req);
  return ret;
}

/* ───────── /sys/module/nvidia{,_uvm} initstate fakes ───────── */

static struct kobject *nvgpu_module_kobj;     /* /sys/module/nvidia     */
static struct kobject *nvgpu_uvm_module_kobj; /* /sys/module/nvidia_uvm */

static ssize_t initstate_show(struct kobject *kobj, struct kobj_attribute *attr,
                              char *buf) {
  return sysfs_emit(buf, "live\n");
}

static struct kobj_attribute initstate_attr = __ATTR_RO(initstate);

static void nvgpu_module_sysfs_init(struct nvgpu_device *dev) {
  struct kobject *modules_kobj;

  /* /sys/module/ is the parent of all module kobjects */
  modules_kobj = kset_find_obj(module_kset, "nvidia");
  if (modules_kobj) {
    /* nvidia.ko already loaded somehow — don't duplicate */
    kobject_put(modules_kobj);
    return;
  }

  nvgpu_module_kobj = kobject_create_and_add("nvidia", &module_kset->kobj);
  if (!nvgpu_module_kobj) {
    dev_warn(&dev->vdev->dev,
             "virtio-gpu-nv: failed to create /sys/module/nvidia\n");
    return;
  }
  if (sysfs_create_file(nvgpu_module_kobj, &initstate_attr.attr))
    dev_warn(&dev->vdev->dev, "virtio-gpu-nv: failed initstate under nvidia\n");

  nvgpu_uvm_module_kobj =
      kobject_create_and_add("nvidia_uvm", &module_kset->kobj);
  if (!nvgpu_uvm_module_kobj) {
    dev_warn(&dev->vdev->dev,
             "virtio-gpu-nv: failed to create /sys/module/nvidia_uvm\n");
    return;
  }
  if (sysfs_create_file(nvgpu_uvm_module_kobj, &initstate_attr.attr))
    dev_warn(&dev->vdev->dev,
             "virtio-gpu-nv: failed initstate under nvidia_uvm\n");

  dev_info(&dev->vdev->dev,
           "virtio-gpu-nv: created /sys/module/nvidia{,_uvm}/initstate\n");
}

static void nvgpu_module_sysfs_cleanup(void) {
  if (nvgpu_uvm_module_kobj) {
    sysfs_remove_file(nvgpu_uvm_module_kobj, &initstate_attr.attr);
    kobject_put(nvgpu_uvm_module_kobj);
    nvgpu_uvm_module_kobj = NULL;
  }
  if (nvgpu_module_kobj) {
    sysfs_remove_file(nvgpu_module_kobj, &initstate_attr.attr);
    kobject_put(nvgpu_module_kobj);
    nvgpu_module_kobj = NULL;
  }
}

/* ── nvidia-caps fops — proxy to host like everything else ── */

static int nvgpu_caps_open(struct inode *inode, struct file *filp) {
  /* caps devices are read-only capability checks.
   * NVIDIA userspace opens them, does a few ioctls, closes.
   * For now return success with a NULL private_data —
   * if actual ioctls are needed we'll add VMM proxying. */
  filp->private_data = NULL;
  return 0;
}

static int nvgpu_caps_release(struct inode *inode, struct file *filp) {
  return 0;
}

static long nvgpu_caps_ioctl(struct file *filp, unsigned int cmd,
                             unsigned long arg) {
  /* Most caps ioctls just query capability bits.
   * Return 0 (success) — tells userspace "no special capabilities"
   * which is correct for a non-MIG single GPU. */
  return 0;
}

static const struct file_operations nvgpu_caps_fops = {
    .owner = THIS_MODULE,
    .open = nvgpu_caps_open,
    .release = nvgpu_caps_release,
    .unlocked_ioctl = nvgpu_caps_ioctl,
};

static struct class *nvgpu_caps_class;

static char *nvgpu_caps_devnode(const struct device *dev, umode_t *mode) {
  if (mode)
    *mode = 0444;
  return kasprintf(GFP_KERNEL, "nvidia-caps/%s", dev_name(dev));
}

/* ───────── Probe / remove ───────── */

static int nvgpu_probe(struct virtio_device *vdev) {
  struct nvgpu_device *dev;
  struct virtqueue_info vqs_info[] = {
      {"control", nvgpu_ctrl_vq_cb},
      {"event", nvgpu_event_vq_cb},
  };
  struct virtqueue *vqs[2];
  dev_t gpu_devno;
  int ret, i;

  dev = devm_kzalloc(&vdev->dev, sizeof(*dev), GFP_KERNEL);
  if (!dev)
    return -ENOMEM;

  dev->vdev = vdev;
  vdev->priv = dev;
  mutex_init(&dev->vq_lock);
  init_completion(&dev->req_done);

  /* Find virtqueues */
  ret = virtio_find_vqs(vdev, 2, vqs, vqs_info, NULL);
  if (ret)
    return ret;

  dev->ctrl_vq = vqs[0];
  dev->event_vq = vqs[1];

  /* Read config space written by the VMM at device creation */
  virtio_cread_bytes(vdev, 0, dev->driver_version, 32);
  dev->driver_version[31] = '\0';
  virtio_cread(vdev, struct virtio_gpu_nv_config, num_gpus, &dev->num_gpus);
  virtio_cread(vdev, struct virtio_gpu_nv_config, caps, &dev->caps);

  if (dev->num_gpus == 0 || dev->num_gpus > 248) {
    dev_err(&vdev->dev, "virtio-gpu-nv: bad num_gpus %u\n", dev->num_gpus);
    return -EINVAL;
  }

  /* GPU info records */
  {
    u32 i;
    for (i = 0; i < dev->num_gpus && i < 8; i++) {
      size_t off = offsetof(struct virtio_gpu_nv_config, gpus[i]);
      virtio_cread_bytes(vdev, off, &dev->gpu_slots[i],
                         sizeof(dev->gpu_slots[i]));

      /* Safety: ensure pci_addr is NUL-terminated before logging */
      dev->gpu_slots[i].pci_addr[15] = '\0';

      dev_info(&vdev->dev,
               "virtio-gpu-nv: GPU%u  pci=%s  minor=%u  info_len=%u\n", i,
               dev->gpu_slots[i].pci_addr, le32_to_cpu(dev->gpu_slots[i].minor),
               le32_to_cpu(dev->gpu_slots[i].info_len));
    }
  }

  /* FD translation table */
  virtio_cread(vdev, struct virtio_gpu_nv_config, num_fd_translations,
               &dev->num_fd_translations);

  if (dev->num_fd_translations > 16)
    dev->num_fd_translations = 16;

  if (dev->num_fd_translations > 0) {
    size_t off = offsetof(struct virtio_gpu_nv_config, fd_translations);
    virtio_cread_bytes(vdev, off, dev->fd_translations,
                       dev->num_fd_translations *
                           sizeof(dev->fd_translations[0]));
  }

  dev_info(&vdev->dev, "virtio-gpu-nv: %u fd-translation ioctl(s) registered\n",
           dev->num_fd_translations);

  /* Ensure virtio is running before we open devices */
  virtio_device_ready(vdev);

  /* Create device class once */
  nvgpu_class = class_create("nvidia");
  if (IS_ERR(nvgpu_class)) {
    ret = PTR_ERR(nvgpu_class);
    nvgpu_class = NULL;
    return ret;
  }

  /* Set more open permissions to device node */
  nvgpu_class->devnode = nvgpu_devnode;

  /* Register /dev/nvidia0 … /dev/nvidia<N-1> */
  gpu_devno = MKDEV(NV_MAJOR, 0);
  ret = register_chrdev_region(gpu_devno, dev->num_gpus, "nvidia");
  if (ret)
    goto err_class;

  for (i = 0; i < (int)dev->num_gpus; i++) {
    cdev_init(&dev->cdev_gpu[i], &nvgpu_gpu_fops);
    dev->cdev_gpu[i].owner = THIS_MODULE;
    ret = cdev_add(&dev->cdev_gpu[i], MKDEV(NV_MAJOR, i), 1);
    if (ret)
      goto err_gpu_cdevs;
    device_create(nvgpu_class, &vdev->dev, MKDEV(NV_MAJOR, i), NULL, "nvidia%d",
                  i);
  }

  /* Register /dev/nvidiactl */
  ret = register_chrdev_region(MKDEV(NV_MAJOR, NV_CTL_MINOR), 1, "nvidiactl");
  if (ret)
    goto err_gpu_cdevs;

  cdev_init(&dev->cdev_ctl, &nvgpu_ctl_fops);
  dev->cdev_ctl.owner = THIS_MODULE;
  ret = cdev_add(&dev->cdev_ctl, MKDEV(NV_MAJOR, NV_CTL_MINOR), 1);
  if (ret)
    goto err_ctl_region;

  device_create(nvgpu_class, &vdev->dev, MKDEV(NV_MAJOR, NV_CTL_MINOR), NULL,
                "nvidiactl");

  /* Register /dev/nvidia-uvm (major should match host) */
  dev->uvm_devno = MKDEV(NV_UVM_MAJOR, 0);
  ret = register_chrdev_region(dev->uvm_devno, 2, "nvidia-uvm");
  if (ret)
    goto err_ctl_cdev;

  cdev_init(&dev->cdev_uvm, &nvgpu_uvm_fops);
  dev->cdev_uvm.owner = THIS_MODULE;
  ret = cdev_add(&dev->cdev_uvm, dev->uvm_devno, 1);
  if (ret)
    goto err_uvm_region;

  device_create(nvgpu_class, &vdev->dev, dev->uvm_devno, NULL, "nvidia-uvm");
  device_create(nvgpu_class, &vdev->dev, MKDEV(NV_UVM_MAJOR, 1), NULL,
                "nvidia-uvm-tools");

  /* Register /dev/nvidia-modeset (match host, major 195, minor 254) */
  dev->modeset_devno = MKDEV(NV_MAJOR, NV_MODESET_MINOR);
  ret = register_chrdev_region(dev->modeset_devno, 1, "nvidia-modeset");
  if (ret)
    goto err_gpu_modeset;

  cdev_init(&dev->cdev_modeset, &nvgpu_modeset_fops);
  dev->cdev_modeset.owner = THIS_MODULE;
  ret = cdev_add(&dev->cdev_modeset, dev->modeset_devno, 1);
  if (ret)
    goto err_gpu_modeset;

  device_create(nvgpu_class, &vdev->dev, dev->modeset_devno, NULL,
                "nvidia-modeset");
  dev_info(&vdev->dev,
           "virtio-gpu-nv: registered /dev/nvidia-modeset (%u:%u)\n",
           MAJOR(dev->modeset_devno), MINOR(dev->modeset_devno));

  /* Register /dev/nvidia-caps/nvidia-cap{1,2} */
  dev->caps_devno = MKDEV(NV_CAPS_MAJOR, 1);
  ret = register_chrdev_region(dev->caps_devno, 2, "nvidia-caps");
  if (ret) {
    dev_warn(&vdev->dev, "virtio-gpu-nv: cannot register nvidia-caps: %d\n",
             ret);
    /* non-fatal — continue without caps */
  } else {
    nvgpu_caps_class = class_create("nvidia-caps");
    if (!IS_ERR(nvgpu_caps_class)) {
      nvgpu_caps_class->devnode = nvgpu_caps_devnode;

      cdev_init(&dev->cdev_caps, &nvgpu_caps_fops);
      dev->cdev_caps.owner = THIS_MODULE;
      if (cdev_add(&dev->cdev_caps, MKDEV(NV_CAPS_MAJOR, 1), 2) == 0) {
        device_create(nvgpu_caps_class, &vdev->dev, MKDEV(NV_CAPS_MAJOR, 1),
                      NULL, "nvidia-cap1");
        device_create(nvgpu_caps_class, &vdev->dev, MKDEV(NV_CAPS_MAJOR, 2),
                      NULL, "nvidia-cap2");
        dev_info(
            &vdev->dev,
            "virtio-gpu-nv: registered /dev/nvidia-caps/nvidia-cap{1,2}\n");
      }
    }
  }

  /* Create /proc/driver/nvidia/version */
  ret = nvgpu_proc_init(dev);
  if (ret)
    goto err_uvm_cdev;

  /* Fetch host sysfs content + DRI device list from the VMM */
  ret = nvgpu_fetch_sys_files(dev);
  if (ret)
    dev_warn(&vdev->dev, "virtio-gpu-nv: GET_SYS_FILES failed: %d\n", ret);

  /* Register fake PCI devices — creates /sys/bus/pci/devices/<addr>/ */
  ret = nvgpu_pci_init(dev);
  if (ret)
    dev_warn(&vdev->dev, "virtio-gpu-nv: PCI sysfs init failed: %d\n", ret);

  nvgpu_module_sysfs_init(dev);

  /* Create /dev/dri/renderD128 etc. with host major:minor */
  nvgpu_dri_init(dev); /* non-fatal */

  dev_info(&vdev->dev, "virtio-gpu-nv: %u GPU(s), driver %s\n", dev->num_gpus,
           dev->driver_version);
  return 0;

err_gpu_modeset:
  unregister_chrdev_region(dev->modeset_devno, 1);
err_uvm_region:
  unregister_chrdev_region(dev->uvm_devno, 2);
err_uvm_cdev:
err_ctl_cdev:
  cdev_del(&dev->cdev_ctl);
  device_destroy(nvgpu_class, MKDEV(NV_MAJOR, NV_CTL_MINOR));
err_ctl_region:
  unregister_chrdev_region(MKDEV(NV_MAJOR, NV_CTL_MINOR), 1);
err_gpu_cdevs:
  for (i = i - 1; i >= 0; i--) {
    cdev_del(&dev->cdev_gpu[i]);
    device_destroy(nvgpu_class, MKDEV(NV_MAJOR, i));
  }
  unregister_chrdev_region(MKDEV(NV_MAJOR, 0), dev->num_gpus);
err_class:
  class_destroy(nvgpu_class);
  nvgpu_class = NULL;
  return ret;
}

static void nvgpu_remove(struct virtio_device *vdev) {
  struct nvgpu_device *dev = vdev->priv;
  int i;

  vdev->config->reset(vdev);

  nvgpu_dri_cleanup(dev);
  nvgpu_module_sysfs_cleanup();
  nvgpu_pci_cleanup(dev);

  device_destroy(nvgpu_class, dev->modeset_devno);
  cdev_del(&dev->cdev_modeset);
  unregister_chrdev_region(dev->modeset_devno, 1);

  for (i = 0; i < (int)dev->num_gpus; i++) {
    device_destroy(nvgpu_class, MKDEV(NV_MAJOR, i));
    cdev_del(&dev->cdev_gpu[i]);
  }
  unregister_chrdev_region(MKDEV(NV_MAJOR, 0), dev->num_gpus);

  device_destroy(nvgpu_class, MKDEV(NV_MAJOR, NV_CTL_MINOR));
  cdev_del(&dev->cdev_ctl);
  unregister_chrdev_region(MKDEV(NV_MAJOR, NV_CTL_MINOR), 1);

  device_destroy(nvgpu_class, dev->uvm_devno);
  device_destroy(nvgpu_class, MKDEV(NV_UVM_MAJOR, 1));
  cdev_del(&dev->cdev_uvm);
  unregister_chrdev_region(dev->uvm_devno, 2);

  /* nvidia-caps cleanup */
  if (nvgpu_caps_class) {
    device_destroy(nvgpu_caps_class, MKDEV(NV_CAPS_MAJOR, 1));
    device_destroy(nvgpu_caps_class, MKDEV(NV_CAPS_MAJOR, 2));
    cdev_del(&dev->cdev_caps);
    unregister_chrdev_region(MKDEV(NV_CAPS_MAJOR, 1), 2);
    class_destroy(nvgpu_caps_class);
    nvgpu_caps_class = NULL;
  }

  if (nvgpu_class) {
    class_destroy(nvgpu_class);
    nvgpu_class = NULL;
  }

  vdev->config->del_vqs(vdev);

  remove_proc_subtree("driver/nvidia", NULL);
}

/* ───────── Module boilerplate ───────── */

static struct virtio_device_id id_table[] = {
    {VIRTIO_ID_GPU_NV, VIRTIO_DEV_ANY_ID},
    {0},
};
MODULE_DEVICE_TABLE(virtio, id_table);

static unsigned int features[] = {
    VIRTIO_F_VERSION_1,
    VIRTIO_GPU_NV_F_UVM,
    VIRTIO_GPU_NV_F_ENCODE,
    VIRTIO_GPU_NV_F_GRAPHICS,
};

static struct virtio_driver nvgpu_driver = {
    .driver.name = "virtio-gpu-nv",
    .driver.owner = THIS_MODULE,
    .id_table = id_table,
    .feature_table = features,
    .feature_table_size = ARRAY_SIZE(features),
    .probe = nvgpu_probe,
    .remove = nvgpu_remove,
};

module_virtio_driver(nvgpu_driver);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("libkrun-nv contributors");
MODULE_DESCRIPTION("virtio-gpu-nv: NVIDIA GPU sharing for VMs via ioctl proxy");
