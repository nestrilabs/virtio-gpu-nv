/* SPDX-License-Identifier: Apache-2.0 */
/*
 * virtio_gpu_nv_priv.h — internal data structures for the guest driver.
 *
 * NOT part of the shared guest↔host protocol.  Kernel-internal only.
 */

#ifndef VIRTIO_GPU_NV_PRIV_H
#define VIRTIO_GPU_NV_PRIV_H

#include <linux/atomic.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/mutex.h>
#include <linux/virtio.h>
#include <linux/wait.h>

#include "virtio_gpu_nv.h"

/* Size of the shared bounce buffer (request + response). */
#define NV_BUF_SIZE (2 * (sizeof(struct msg_header) + NV_MAX_PARAM_SIZE))

/* -------------------------------------------------------------------------
 * nv_cdev — one character device minor
 * ---------------------------------------------------------------------- */

struct nv_dev; /* forward */

struct nv_cdev {
  struct cdev cdev;
  struct device *device;
  int minor;           /* our minor index (see MINOR_CTL, etc.) */
  struct nv_dev *ndev; /* back-pointer to parent */
};

/* -------------------------------------------------------------------------
 * nv_dev — one virtio device instance
 * ---------------------------------------------------------------------- */

struct nv_dev {
  struct virtio_device *vdev;

  /* Single virtqueue — all operations are serialised through vq_lock. */
  struct virtqueue *vq;
  struct mutex vq_lock;

  /* Waitqueue: processes sleeping in nv_do_request() wait here. */
  wait_queue_head_t resp_wq;

  /* Bounce buffer: request goes in [0..NV_BUF_SIZE/2),
   *                response goes in [NV_BUF_SIZE/2..NV_BUF_SIZE). */
  void *buf;

  /* Monotonically increasing cookie counter. */
  atomic_t next_cookie;

  /* Array of registered char devices. */
  struct nv_cdev *cdevs;
  int num_cdevs;
};

/* -------------------------------------------------------------------------
 * nv_request — per-operation context (stack-allocated by callers)
 * ---------------------------------------------------------------------- */

struct nv_request {
  /* Filled by nv_do_request() after virtqueue completion. */
  struct resp_header resp_hdr;

  /* Pointer into the response bounce buffer (valid during the call). */
  void *resp_payload;
  u32 resp_payload_len;
};

/* -------------------------------------------------------------------------
 * Function declarations
 * ---------------------------------------------------------------------- */

/* virtio_gpu_nv_vq.c */
void nv_vq_callback(struct virtqueue *vq);

int nv_do_request(struct nv_dev *ndev, const void *req_hdr, size_t req_hdr_size,
                  const void *req_payload, size_t req_payload_size,
                  struct nv_request *out);

/* virtio_gpu_nv_ioctl.c */
long nv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg);

/* virtio_gpu_nv_mmap.c */
int nv_mmap(struct file *filp, struct vm_area_struct *vma);

#endif /* VIRTIO_GPU_NV_PRIV_H */
