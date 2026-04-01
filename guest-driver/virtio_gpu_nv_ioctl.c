// SPDX-License-Identifier: Apache-2.0
/*
 * virtio_gpu_nv_ioctl.c — file operations: open, release, ioctl
 *
 * open():
 *   Sends NV_MSG_OPEN to the backend, receives a guest_handle, stores it in
 *   file->private_data.
 *
 * release():
 *   Sends NV_MSG_CLOSE to the backend.
 *
 * ioctl():
 *   Copies the raw ioctl param bytes from userspace, sends NV_MSG_IOCTL,
 *   waits for the response, copies updated bytes back.  NOT ABI-aware —
 *   the guest driver forwards raw bytes; the backend does all interpretation.
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

  /* Determine device kind from minor. */
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
  if (ret) {
    pr_warn("nv_release: nv_do_request failed: %d\n", ret);
    /* Fall through — still free ctx. */
  } else if (le32_to_cpu(resp.resp_hdr.status) != NV_STATUS_OK) {
    pr_warn("nv_release: backend status %u\n",
            le32_to_cpu(resp.resp_hdr.status));
  }

  kfree(ctx);
  filp->private_data = NULL;
  return 0;
}

/* -------------------------------------------------------------------------
 * nv_ioctl — forward raw ioctl bytes to the backend
 *
 * The guest driver is deliberately NOT ABI-aware.  It:
 *   1. Reads _IOC_SIZE(cmd) bytes from userspace.
 *   2. Sends them to the backend (which is ABI-aware).
 *   3. Copies the (possibly modified) bytes back.
 *
 * The backend returns the updated param bytes alongside the response header.
 * ---------------------------------------------------------------------- */

long nv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
  struct nv_file_ctx *ctx = filp->private_data;
  struct nv_dev *ndev = ctx->dev;
  unsigned int param_size = _IOC_SIZE(cmd);

  /* Stack-allocated request: msg_header + ioctl_req + param bytes. */
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

  /* Build the IOCTL request.
   * We send: [msg_header][ioctl_req][raw param bytes]
   * as a two-part payload: the fixed headers as req_hdr, the param
   * bytes as req_payload.  nv_do_request() concatenates them. */

  req.req_hdr.msg_type = cpu_to_le32(NV_MSG_IOCTL);
  req.req_hdr.cookie = cpu_to_le64(next_cookie(ndev));
  req.req_hdr._pad = 0;

  req.ioctl_hdr.guest_handle = cpu_to_le64(ctx->guest_handle);
  req.ioctl_hdr.request = cpu_to_le64((u64)cmd);
  req.ioctl_hdr.param_size = cpu_to_le32(param_size);
  req.ioctl_hdr._pad = 0;

  /* We need to send [req_hdr + ioctl_req] as a single readable region
   * followed by [param bytes].  nv_do_request takes exactly two regions,
   * so pass the full fixed part as req_hdr and param_buf as req_payload. */
  ret = nv_do_request(ndev, &req, sizeof(req), param_buf, param_size, &resp);
  if (ret)
    goto out;

  if (le32_to_cpu(resp.resp_hdr.status) != NV_STATUS_OK) {
    /* Translate backend status to an appropriate errno. */
    int host_errno = le32_to_cpu(resp.resp_hdr.errno_host);
    ret = host_errno ? -host_errno : -EIO;
    goto out;
  }

  if (resp.resp_payload_len < sizeof(struct ioctl_resp)) {
    ret = -EIO;
    goto out;
  }

  iresp = (struct ioctl_resp *)resp.resp_payload;

  /* Copy updated param bytes back to userspace. */
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
  /* Always report ready for now; NVIDIA user-mode libs will query
   * GPU events via NV_ESC_ALLOC_OS_EVENT/NV_ESC_FREE_OS_EVENT in
   * Phase 4. */
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
