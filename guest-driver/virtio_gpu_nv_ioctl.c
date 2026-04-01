// SPDX-License-Identifier: Apache-2.0
/*
 * virtio_gpu_nv_ioctl.c — file operations: open, release, ioctl
 *
 * open():
 *   Sends NV_MSG_OPEN to the backend, receives a guest_handle, stores it
 *   in file->private_data.
 *
 * release():
 *   Sends NV_MSG_CLOSE to the backend.
 *
 * ioctl():
 *   Copies the raw ioctl param bytes from userspace, sends NV_MSG_IOCTL,
 *   waits for the response, copies updated bytes back.
 *
 * --- On "NOT ABI-aware" ---
 *
 * The guest driver does not parse or interpret any NVIDIA ioctl struct
 * fields.  All semantic logic (handle translation, struct layout,
 * nested-dispatch) lives in the backend.
 *
 * However, the guest driver does use _IOC_SIZE(cmd) to determine how many
 * bytes to copy_from_user / copy_to_user.  This means the ioctl numbers
 * in the user-mode library must encode the correct size for the driver
 * version running on the HOST — because the host backend is the one that
 * actually interprets those bytes.
 *
 * In practice this is always true: the user-mode NVIDIA libraries are
 * version-locked to the kernel driver, and both run at the host's version.
 * The guest's copy of libcuda/libvulkan_nvidia is the same binary as on
 * the host (or the same version), so _IOC_SIZE values match.
 *
 * If you ever load a different-version user-mode library in the guest, the
 * backend will catch the mismatch via NV_ESC_CHECK_VERSION_STR and return
 * an error before any struct is misinterpreted.  We do NOT need to validate
 * _IOC_SIZE in the guest driver beyond the NV_MAX_PARAM_SIZE safety cap.
 *
 * mmap() is in virtio_gpu_nv_mmap.c.
 */

#include <linux/atomic.h>
#include <linux/fs.h>
#include <linux/slab.h>
#include <linux/uaccess.h>

#include "virtio_gpu_nv.h"
#include "virtio_gpu_nv_priv.h"

/* -------------------------------------------------------------------------
 * Cookie generation
 * ---------------------------------------------------------------------- */

static u64 next_cookie(struct nv_dev *ndev) {
  return (u64)atomic_inc_return(&ndev->next_cookie);
}

/* -------------------------------------------------------------------------
 * nv_open — called when userspace opens /dev/nvidia*
 * ---------------------------------------------------------------------- */

static int nv_open(struct inode *inode, struct file *filp) {
  struct nv_cdev *ncdev = container_of(inode->i_cdev, struct nv_cdev, cdev);
  struct nv_dev *ndev = ncdev->ndev;
  struct nv_file_ctx *ctx;
  struct msg_header req_hdr;
  struct open_req req_payload;
  struct nv_request resp;
  struct open_resp *oresp;
  int ret;

  ctx = kzalloc(sizeof(*ctx), GFP_KERNEL);
  if (!ctx)
    return -ENOMEM;

  ctx->dev = ndev;

  memset(&req_payload, 0, sizeof(req_payload));
  if (ncdev->minor == MINOR_CTL) {
    req_payload.kind = NV_DEV_CTL;
    req_payload.index = 0;
  } else if (ncdev->minor == MINOR_UVM) {
    req_payload.kind = NV_DEV_UVM;
    req_payload.index = 0;
  } else {
    req_payload.kind = NV_DEV_GPU;
    req_payload.index = (u8)(ncdev->minor - MINOR_GPU_BASE);
  }

  req_hdr.msg_type = cpu_to_le32(NV_MSG_OPEN);
  req_hdr.cookie = cpu_to_le64(next_cookie(ndev));
  req_hdr._pad = 0;

  ret = nv_do_request(ndev, &req_hdr, sizeof(req_hdr), &req_payload,
                      sizeof(req_payload), &resp);
  if (ret)
    goto err_free;

  if (le32_to_cpu(resp.resp_hdr.status) != NV_STATUS_OK) {
    pr_err("nv_open: backend returned status %u (host errno %d)\n",
           le32_to_cpu(resp.resp_hdr.status),
           le32_to_cpu(resp.resp_hdr.errno_host));
    ret = -EIO;
    goto err_free;
  }

  if (resp.resp_payload_len < sizeof(struct open_resp)) {
    pr_err("nv_open: response payload too short\n");
    ret = -EIO;
    goto err_free;
  }

  oresp = (struct open_resp *)resp.resp_payload;
  ctx->guest_handle = le64_to_cpu(oresp->guest_handle);

  filp->private_data = ctx;
  return 0;

err_free:
  kfree(ctx);
  return ret;
}

/* -------------------------------------------------------------------------
 * nv_release — called when the last fd reference is dropped
 * ---------------------------------------------------------------------- */

static int nv_release(struct inode *inode, struct file *filp) {
  struct nv_file_ctx *ctx = filp->private_data;
  struct nv_dev *ndev = ctx->dev;
  struct msg_header req_hdr;
  struct close_req req_payload;
  struct nv_request resp;
  int ret;

  req_hdr.msg_type = cpu_to_le32(NV_MSG_CLOSE);
  req_hdr.cookie = cpu_to_le64(next_cookie(ndev));
  req_hdr._pad = 0;

  req_payload.guest_handle = cpu_to_le64(ctx->guest_handle);

  ret = nv_do_request(ndev, &req_hdr, sizeof(req_hdr), &req_payload,
                      sizeof(req_payload), &resp);
  if (ret)
    pr_warn("nv_release: nv_do_request failed: %d\n", ret);
  else if (le32_to_cpu(resp.resp_hdr.status) != NV_STATUS_OK)
    pr_warn("nv_release: backend status %u\n",
            le32_to_cpu(resp.resp_hdr.status));

  kfree(ctx);
  filp->private_data = NULL;
  return 0;
}

/* -------------------------------------------------------------------------
 * nv_ioctl — forward raw ioctl bytes to the backend
 *
 * Size source: _IOC_SIZE(cmd) — the size encoded by the user-mode library
 * in the ioctl number.  See the file-level comment for why this is correct.
 * ---------------------------------------------------------------------- */

long nv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
  struct nv_file_ctx *ctx = filp->private_data;
  struct nv_dev *ndev = ctx->dev;
  unsigned int param_size = _IOC_SIZE(cmd);

  struct {
    struct msg_header req_hdr;
    struct ioctl_req ioctl_hdr;
  } req;

  struct nv_request resp;
  struct ioctl_resp *iresp;
  void *param_buf = NULL;
  long ret;

  if (param_size > NV_MAX_PARAM_SIZE)
    return -EINVAL;

  if (param_size) {
    param_buf = kmalloc(param_size, GFP_KERNEL);
    if (!param_buf)
      return -ENOMEM;

    if (copy_from_user(param_buf, (void __user *)arg, param_size)) {
      ret = -EFAULT;
      goto out;
    }
  }

  req.req_hdr.msg_type = cpu_to_le32(NV_MSG_IOCTL);
  req.req_hdr.cookie = cpu_to_le64(next_cookie(ndev));
  req.req_hdr._pad = 0;

  req.ioctl_hdr.guest_handle = cpu_to_le64(ctx->guest_handle);
  req.ioctl_hdr.request = cpu_to_le64((u64)cmd);
  req.ioctl_hdr.param_size = cpu_to_le32(param_size);
  req.ioctl_hdr._pad = 0;

  ret = nv_do_request(ndev, &req, sizeof(req), param_buf, param_size, &resp);
  if (ret)
    goto out;

  if (le32_to_cpu(resp.resp_hdr.status) != NV_STATUS_OK) {
    int host_errno = le32_to_cpu(resp.resp_hdr.errno_host);
    ret = host_errno ? -host_errno : -EIO;
    goto out;
  }

  if (resp.resp_payload_len < sizeof(struct ioctl_resp)) {
    ret = -EIO;
    goto out;
  }

  iresp = (struct ioctl_resp *)resp.resp_payload;

  if (param_size && iresp->param_size) {
    u32 copy_len = min_t(u32, param_size, le32_to_cpu(iresp->param_size));
    void *resp_params = (char *)resp.resp_payload + sizeof(struct ioctl_resp);

    if (copy_to_user((void __user *)arg, resp_params, copy_len)) {
      ret = -EFAULT;
      goto out;
    }
  }

  ret = 0;
out:
  kfree(param_buf);
  return ret;
}

/* -------------------------------------------------------------------------
 * poll — stub for Phase 1; Phase 4 will implement GPU event polling
 * ---------------------------------------------------------------------- */

static __poll_t nv_poll(struct file *filp, struct poll_table_struct *wait) {
  return EPOLLIN | EPOLLOUT;
}

/* -------------------------------------------------------------------------
 * File operations table
 * ---------------------------------------------------------------------- */

const struct file_operations nv_fops = {
    .owner = THIS_MODULE,
    .open = nv_open,
    .release = nv_release,
    .unlocked_ioctl = nv_ioctl,
    .compat_ioctl = nv_ioctl,
    .mmap = nv_mmap,
    .poll = nv_poll,
};
