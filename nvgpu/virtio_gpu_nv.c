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

#include <linux/cdev.h>
#include <linux/completion.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/proc_fs.h>
#include <linux/scatterlist.h>
#include <linux/seq_file.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/virtio.h>
#include <linux/virtio_config.h>
#include <linux/virtio_ids.h>

/* ───────── virtio device identity ───────── */

#define VIRTIO_ID_GPU_NV 45

/* Feature bits */
#define VIRTIO_GPU_NV_F_UVM 0
#define VIRTIO_GPU_NV_F_ENCODE 1
#define VIRTIO_GPU_NV_F_GRAPHICS 2

/* ───────── NVIDIA device node numbers ───────── */

#define NV_MAJOR 195
#define NV_CTL_MINOR 255

/* ───────── Wire protocol constants ───────── */

#define NVGPU_MSG_OPEN 1
#define NVGPU_MSG_CLOSE 2
#define NVGPU_MSG_IOCTL 3
#define NVGPU_MSG_MMAP 4
#define NVGPU_MSG_MUNMAP 5

/* device_type values for OPEN */
#define NVGPU_DEV_CTL 255
#define NVGPU_DEV_UVM 256
#define NVGPU_DEV_UVM_TOOLS 257
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

/* VMM config space layout */
struct virtio_gpu_nv_config {
  char driver_version[32];
  __le32 num_gpus;
  __le32 caps;
  __le32 gpu_device_ids[8];
} __packed;

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

/* Forward declarations */
struct nvgpu_device;
struct nvgpu_fd;

/*
 * Per-device state — one instance per virtio device probe.
 */
struct nvgpu_device {
  struct virtio_device *vdev;
  struct virtqueue *ctrl_vq;
  struct virtqueue *event_vq;

  /* Character device registration */
  struct cdev cdev_gpu[248]; /* /dev/nvidia0 … nvidia247 */
  struct cdev cdev_ctl;      /* /dev/nvidiactl            */
  struct cdev cdev_uvm;      /* /dev/nvidia-uvm           */
  dev_t uvm_devno;           /* dynamic major for UVM     */

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

  virtqueue_kick(dev->ctrl_vq);
  mutex_unlock(&dev->vq_lock);

  wait_for_completion(&dev->req_done);
  return 0;
}

/* Virtqueue callback: VMM has written the response buffer */
static void nvgpu_ctrl_vq_cb(struct virtqueue *vq) {
  struct nvgpu_device *dev = vq->vdev->priv;
  void *buf;
  unsigned int len;

  while ((buf = virtqueue_get_buf(vq, &len)) != NULL)
    complete(&dev->req_done);
}

/* event virtqueue callback — not used yet, just drain */
static void nvgpu_event_vq_cb(struct virtqueue *vq) {
  void *buf;
  unsigned int len;

  while ((buf = virtqueue_get_buf(vq, &len)) != NULL)
    /* TODO: deliver to waiting guest processes */;
}

/* ───────── Ioctl forwarding ───────── */

/*
 * nvgpu_ioctl_simple — flat struct, no embedded pointers.
 * Covers ~95 % of NVIDIA ioctls.
 */
static long nvgpu_ioctl_simple(struct nvgpu_fd *nfd, unsigned int cmd,
                               void __user *uarg, unsigned int sz) {
  int req_total = sizeof(struct nvgpu_ioctl_req) + sz;
  int resp_max = sizeof(struct nvgpu_ioctl_resp) + sz;
  void *req_buf, *resp_buf;
  struct nvgpu_ioctl_req *req;
  struct nvgpu_ioctl_resp *resp;
  int ret;

  req_buf = kvmalloc(req_total, GFP_KERNEL);
  resp_buf = kvmalloc(resp_max, GFP_KERNEL);
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

  if (copy_from_user(req_buf + sizeof(*req), uarg, sz)) {
    ret = -EFAULT;
    goto out;
  }

  ret = nvgpu_send_recv(nfd->dev, req_buf, req_total, resp_buf, resp_max);
  if (ret < 0)
    goto out;

  resp = (struct nvgpu_ioctl_resp *)resp_buf;
  ret = (int)le32_to_cpu((__le32)resp->hdr.status);

  if (resp->data_len && le32_to_cpu(resp->data_len) <= sz) {
    if (copy_to_user(uarg, resp_buf + sizeof(*resp),
                     le32_to_cpu(resp->data_len)))
      ret = -EFAULT;
  }

out:
  kvfree(req_buf);
  kvfree(resp_buf);
  return ret;
}

/*
 * nvgpu_ioctl_rm_control — NV_ESC_RM_CONTROL has a nested params buffer
 * addressed via NVOS54_PARAMETERS.params.
 */
static long nvgpu_ioctl_rm_control(struct nvgpu_fd *nfd, unsigned int cmd,
                                   void __user *uarg, unsigned int sz) {
  struct NVOS54_PARAMETERS params;
  void __user *user_nested;
  void *req_buf = NULL, *resp_buf = NULL, *nested;
  struct nvgpu_ioctl_req *req;
  struct nvgpu_ioctl_resp *resp;
  int req_total, resp_max, ret;

  if (sz < sizeof(params))
    return -EINVAL;

  if (copy_from_user(&params, uarg, sizeof(params)))
    return -EFAULT;

  user_nested = (void __user *)(unsigned long)le64_to_cpu(params.params);

  if (le32_to_cpu(params.paramsSize) > 1024 * 1024)
    return -EINVAL;

  req_total = sizeof(*req) + sizeof(params) + le32_to_cpu(params.paramsSize);
  resp_max = sizeof(struct nvgpu_ioctl_resp) + sizeof(params) +
             le32_to_cpu(params.paramsSize);

  req_buf = kvmalloc(req_total, GFP_KERNEL);
  resp_buf = kvmalloc(resp_max, GFP_KERNEL);
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
  req->nested_len = params.paramsSize; /* already LE */

  /* Copy top-level struct (guest pointer left in place; VMM ignores it) */
  memcpy(req_buf + sizeof(*req), &params, sizeof(params));

  /* Copy nested data from guest userspace */
  if (user_nested && le32_to_cpu(params.paramsSize) > 0) {
    nested = req_buf + sizeof(*req) + sizeof(params);
    if (copy_from_user(nested, user_nested, le32_to_cpu(params.paramsSize))) {
      ret = -EFAULT;
      goto out;
    }
  }

  ret = nvgpu_send_recv(nfd->dev, req_buf, req_total, resp_buf, resp_max);
  if (ret < 0)
    goto out;

  resp = (struct nvgpu_ioctl_resp *)resp_buf;
  ret = (int)(s32)le32_to_cpu((__le32)resp->hdr.status);

  /* Copy modified top-level struct back */
  if (copy_to_user(uarg, resp_buf + sizeof(*resp), sizeof(params))) {
    ret = -EFAULT;
    goto out;
  }

  /* Copy modified nested data back to original guest pointer */
  if (user_nested && le32_to_cpu(resp->nested_len) > 0) {
    if (copy_to_user(user_nested, resp_buf + sizeof(*resp) + sizeof(params),
                     le32_to_cpu(resp->nested_len)))
      ret = -EFAULT;
  }

out:
  kvfree(req_buf);
  kvfree(resp_buf);
  return ret;
}

/*
 * nvgpu_ioctl_rm_alloc — NV_ESC_RM_ALLOC, same pattern via NVOS64_PARAMETERS.
 */
static long nvgpu_ioctl_rm_alloc(struct nvgpu_fd *nfd, unsigned int cmd,
                                 void __user *uarg, unsigned int sz) {
  struct NVOS64_PARAMETERS params;
  void __user *user_alloc;
  void *req_buf = NULL, *resp_buf = NULL, *nested;
  struct nvgpu_ioctl_req *req;
  struct nvgpu_ioctl_resp *resp;
  int req_total, resp_max, ret;

  if (sz < sizeof(params))
    return -EINVAL;

  if (copy_from_user(&params, uarg, sizeof(params)))
    return -EFAULT;

  user_alloc = (void __user *)(unsigned long)le64_to_cpu(params.pAllocParms);

  if (le32_to_cpu(params.paramsSize) > 1024 * 1024)
    return -EINVAL;

  req_total = sizeof(*req) + sizeof(params) + le32_to_cpu(params.paramsSize);
  resp_max = sizeof(struct nvgpu_ioctl_resp) + sizeof(params) +
             le32_to_cpu(params.paramsSize);

  req_buf = kvmalloc(req_total, GFP_KERNEL);
  resp_buf = kvmalloc(resp_max, GFP_KERNEL);
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
  req->nested_len = params.paramsSize;

  memcpy(req_buf + sizeof(*req), &params, sizeof(params));

  if (user_alloc && le32_to_cpu(params.paramsSize) > 0) {
    nested = req_buf + sizeof(*req) + sizeof(params);
    if (copy_from_user(nested, user_alloc, le32_to_cpu(params.paramsSize))) {
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
    if (copy_to_user(user_alloc, resp_buf + sizeof(*resp) + sizeof(params),
                     le32_to_cpu(resp->nested_len)))
      ret = -EFAULT;
  }

out:
  kvfree(req_buf);
  kvfree(resp_buf);
  return ret;
}

/* Main ioctl dispatcher */
static long nvgpu_ioctl(struct file *filp, unsigned int cmd,
                        unsigned long arg) {
  struct nvgpu_fd *nfd = filp->private_data;
  unsigned int nr = _IOC_NR(cmd);
  unsigned int sz = _IOC_SIZE(cmd);
  void __user *uarg = (void __user *)arg;

  if (sz == 0 || sz > 4096)
    return -EINVAL;

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

/*
 * UVM ioctls are all flat (no nested pointers), with one exception:
 * UVM_INITIALIZE gets the multi-process sharing flag injected.
 */
static long nvgpu_uvm_ioctl(struct file *filp, unsigned int cmd,
                            unsigned long arg) {
  struct nvgpu_fd *nfd = filp->private_data;
  unsigned int nr = _IOC_NR(cmd);
  unsigned int sz = _IOC_SIZE(cmd);
  void __user *uarg = (void __user *)arg;

  if (sz == 0 || sz > 4096)
    return -EINVAL;

  if (nr == UVM_INITIALIZE_NR) {
    /*
     * Set UVM_INIT_FLAGS_MULTI_PROCESS_SHARING_MODE so the host
     * driver allows this UVM context to be shared across processes
     * (one per VM).  The flag is at offset 0 in UvmInitializeParams.
     * We read, OR the bit, write back, then forward.
     */
    u64 flags;

    if (sz >= sizeof(flags)) {
      if (copy_from_user(&flags, uarg, sizeof(flags)))
        return -EFAULT;
      flags |= (1ULL << 2); /* UVM_INIT_FLAGS_MULTI_PROCESS_SHARING_MODE */
      if (copy_to_user(uarg, &flags, sizeof(flags)))
        return -EFAULT;
    }
  }

  return nvgpu_ioctl_simple(nfd, cmd, uarg, sz);
}

/* ───────── mmap ───────── */

static void nvgpu_vma_close(struct vm_area_struct *vma) {
  struct nvgpu_fd *nfd = vma->vm_file->private_data;
  u32 mapping_id = (u32)(unsigned long)vma->vm_private_data;
  struct nvgpu_munmap_req req = {};
  struct nvgpu_munmap_resp resp;

  req.hdr.msg_type = cpu_to_le32(NVGPU_MSG_MUNMAP);
  req.hdr.handle = cpu_to_le32(nfd->handle);
  req.mapping_id = cpu_to_le32(mapping_id);

  nvgpu_send_recv(nfd->dev, &req, sizeof(req), &resp, sizeof(resp));
}

static const struct vm_operations_struct nvgpu_vm_ops = {
    .close = nvgpu_vma_close,
};

static int nvgpu_mmap(struct file *filp, struct vm_area_struct *vma) {
  struct nvgpu_fd *nfd = filp->private_data;
  u64 size = vma->vm_end - vma->vm_start;
  u64 offset = (u64)vma->vm_pgoff << PAGE_SHIFT;
  struct nvgpu_mmap_req req = {};
  struct nvgpu_mmap_resp resp;
  int ret;

  req.hdr.msg_type = cpu_to_le32(NVGPU_MSG_MMAP);
  req.hdr.handle = cpu_to_le32(nfd->handle);
  req.size = cpu_to_le64(size);
  req.offset = cpu_to_le64(offset);
  req.prot = cpu_to_le32((vma->vm_flags & VM_WRITE) ? 3 : 1);

  ret = nvgpu_send_recv(nfd->dev, &req, sizeof(req), &resp, sizeof(resp));
  if (ret < 0)
    return ret;
  if ((s32)le32_to_cpu((__le32)resp.hdr.status) < 0)
    return (s32)le32_to_cpu((__le32)resp.hdr.status);

  /*
   * VMM has:
   *  1. mmap()'d the host NVIDIA fd.
   *  2. Created a KVM memory slot at resp.guest_phys_addr pointing to
   *     those host pages.
   *
   * Wire that GPA into the guest process's VMA now.
   */
  vm_flags_set(vma, VM_IO | VM_PFNMAP | VM_DONTEXPAND | VM_DONTDUMP);
  vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot);

  ret = remap_pfn_range(vma, vma->vm_start,
                        le64_to_cpu(resp.guest_phys_addr) >> PAGE_SHIFT, size,
                        vma->vm_page_prot);
  if (ret)
    return ret;

  vma->vm_ops = &nvgpu_vm_ops;
  vma->vm_private_data = (void *)(unsigned long)le32_to_cpu(resp.mapping_id);
  return 0;
}

/* ───────── open / release ───────── */

static int nvgpu_open_common(struct inode *inode, struct file *filp,
                             u32 device_type) {
  struct nvgpu_device *dev;
  struct nvgpu_fd *nfd;
  struct nvgpu_open_req req = {};
  struct nvgpu_open_resp resp = {};
  int ret;

  /* Recover nvgpu_device pointer depending on which cdev was opened */
  if (device_type == NVGPU_DEV_UVM || device_type == NVGPU_DEV_UVM_TOOLS)
    dev = container_of(inode->i_cdev, struct nvgpu_device, cdev_uvm);
  else if (device_type == NVGPU_DEV_CTL)
    dev = container_of(inode->i_cdev, struct nvgpu_device, cdev_ctl);
  else
    dev = container_of(inode->i_cdev, struct nvgpu_device,
                       cdev_gpu[iminor(inode)]);

  nfd = kzalloc(sizeof(*nfd), GFP_KERNEL);
  if (!nfd)
    return -ENOMEM;

  nfd->dev = dev;
  nfd->device_type = device_type;

  req.hdr.msg_type = cpu_to_le32(NVGPU_MSG_OPEN);
  req.device_type = cpu_to_le32(device_type);
  req.flags = cpu_to_le32(filp->f_flags);

  ret = nvgpu_send_recv(dev, &req, sizeof(req), &resp, sizeof(resp));
  if (ret < 0 || (s32)le32_to_cpu((__le32)resp.hdr.status) < 0) {
    kfree(nfd);
    if (ret < 0)
      return ret;
    return (s32)le32_to_cpu((__le32)resp.hdr.status);
  }

  nfd->handle = le32_to_cpu(resp.hdr.handle);
  filp->private_data = nfd;
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
  struct nvgpu_msg_hdr req = {
      .msg_type = cpu_to_le32(NVGPU_MSG_CLOSE),
      .handle = cpu_to_le32(nfd->handle),
  };
  struct nvgpu_msg_hdr resp;

  nvgpu_send_recv(nfd->dev, &req, sizeof(req), &resp, sizeof(resp));
  /* VMM tears down all mappings associated with this handle */

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
    .llseek = no_llseek,
};

static const struct file_operations nvgpu_ctl_fops = {
    .owner = THIS_MODULE,
    .open = nvgpu_ctl_open,
    .release = nvgpu_release,
    .unlocked_ioctl = nvgpu_ioctl,
    .mmap = nvgpu_mmap,
    .llseek = no_llseek,
};

static const struct file_operations nvgpu_uvm_fops = {
    .owner = THIS_MODULE,
    .open = nvgpu_uvm_open,
    .release = nvgpu_release,
    .unlocked_ioctl = nvgpu_uvm_ioctl,
    .mmap = nvgpu_mmap,
    .llseek = no_llseek,
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

static int nvgpu_proc_init(struct nvgpu_device *dev) {
  struct proc_dir_entry *nvidia_dir;

  nvidia_dir = proc_mkdir("driver/nvidia", NULL);
  if (!nvidia_dir)
    return -ENOMEM;

  proc_create_data("version", 0444, nvidia_dir, &nvgpu_proc_version_ops, dev);
  proc_create_data("params", 0444, nvidia_dir, &nvgpu_proc_params_ops, dev);
  return 0;
}

/* ───────── Probe / remove ───────── */

static int nvgpu_probe(struct virtio_device *vdev) {
  struct nvgpu_device *dev;
  static const char *vq_names[] = {"control", "event"};
  vq_callback_t *cbs[] = {nvgpu_ctrl_vq_cb, nvgpu_event_vq_cb};
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
  ret = virtio_find_vqs(vdev, 2, vqs, cbs, vq_names, NULL);
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

  /* Ensure virtio is running before we open devices */
  virtio_device_ready(vdev);

  /* Create device class once */
  nvgpu_class = class_create("nvidia");
  if (IS_ERR(nvgpu_class)) {
    ret = PTR_ERR(nvgpu_class);
    nvgpu_class = NULL;
    return ret;
  }

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

  /* Register /dev/nvidia-uvm (dynamic major) */
  ret = alloc_chrdev_region(&dev->uvm_devno, 0, 2, "nvidia-uvm");
  if (ret)
    goto err_ctl_cdev;

  cdev_init(&dev->cdev_uvm, &nvgpu_uvm_fops);
  dev->cdev_uvm.owner = THIS_MODULE;
  ret = cdev_add(&dev->cdev_uvm, dev->uvm_devno, 1);
  if (ret)
    goto err_uvm_region;

  device_create(nvgpu_class, &vdev->dev, dev->uvm_devno, NULL, "nvidia-uvm");
  device_create(nvgpu_class, &vdev->dev, MKDEV(MAJOR(dev->uvm_devno), 1), NULL,
                "nvidia-uvm-tools");

  /* Create /proc/driver/nvidia/version */
  nvgpu_proc_init(dev);

  dev_info(&vdev->dev, "virtio-gpu-nv: %u GPU(s), driver %s\n", dev->num_gpus,
           dev->driver_version);
  return 0;

err_uvm_region:
  unregister_chrdev_region(dev->uvm_devno, 2);
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

  for (i = 0; i < (int)dev->num_gpus; i++) {
    device_destroy(nvgpu_class, MKDEV(NV_MAJOR, i));
    cdev_del(&dev->cdev_gpu[i]);
  }
  unregister_chrdev_region(MKDEV(NV_MAJOR, 0), dev->num_gpus);

  device_destroy(nvgpu_class, MKDEV(NV_MAJOR, NV_CTL_MINOR));
  cdev_del(&dev->cdev_ctl);
  unregister_chrdev_region(MKDEV(NV_MAJOR, NV_CTL_MINOR), 1);

  device_destroy(nvgpu_class, dev->uvm_devno);
  device_destroy(nvgpu_class, MKDEV(MAJOR(dev->uvm_devno), 1));
  cdev_del(&dev->cdev_uvm);
  unregister_chrdev_region(dev->uvm_devno, 2);

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
MODULE_AUTHOR("libkrun contributors");
MODULE_DESCRIPTION("virtio-gpu-nv: NVIDIA GPU sharing for VMs via ioctl proxy");
