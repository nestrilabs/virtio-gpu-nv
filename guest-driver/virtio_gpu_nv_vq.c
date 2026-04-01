// SPDX-License-Identifier: Apache-2.0
/*
 * virtio_gpu_nv_vq.c — virtqueue helpers
 *
 * nv_do_request(): the single entry point for posting a request and waiting
 *                  for the backend to complete it.
 *
 * Design (Phase 1):
 *   We serialise all operations through a single mutex + bounce buffer.
 *   Each call:
 *     1. Assembles the request in the first half of the bounce buffer.
 *     2. Adds two descriptors to the virtqueue: one readable (req), one
 *        writable (resp).
 *     3. Kicks the backend.
 *     4. Sleeps on resp_wq until the used-ring callback fires.
 *     5. Copies the response header and payload pointer into *out.
 *
 * Phase 3 will switch to per-request allocation and a response-cookie
 * hash table so multiple operations can be in-flight simultaneously.
 */

#include <linux/scatterlist.h>
#include <linux/slab.h>
#include <linux/string.h>
#include <linux/virtio.h>
#include <linux/wait.h>

#include "virtio_gpu_nv.h"
#include "virtio_gpu_nv_priv.h"

/* -------------------------------------------------------------------------
 * Completion flag
 *
 * We keep a single per-device flag.  nv_vq_callback() sets it and wakes
 * resp_wq; nv_do_request() clears it before posting and waits for it.
 * This is safe because vq_lock ensures only one request is in-flight.
 * ---------------------------------------------------------------------- */

#define NV_RESP_PENDING 0
#define NV_RESP_DONE 1

/* -------------------------------------------------------------------------
 * nv_vq_callback — called from virtio interrupt context
 * ---------------------------------------------------------------------- */

void nv_vq_callback(struct virtqueue *vq) {
  struct nv_dev *ndev = vq->vdev->priv;

  /* The used-ring has been updated; wake anyone sleeping in
   * nv_do_request().  The actual used-buffer retrieval happens there. */
  wake_up(&ndev->resp_wq);
}

/* -------------------------------------------------------------------------
 * nv_do_request — post one request and wait for the response
 *
 * Caller must NOT hold vq_lock.  We acquire it here.
 *
 * @req_hdr:          Pointer to the msg_header (already filled in).
 * @req_hdr_size:     sizeof(msg_header).
 * @req_payload:      Pointer to the message-specific request payload,
 *                    or NULL if there is none.
 * @req_payload_size: Size of the request payload in bytes.
 * @out:              Filled with the parsed response on success.
 *
 * Returns 0 on success, -errno on error.
 * ---------------------------------------------------------------------- */

int nv_do_request(struct nv_dev *ndev, const void *req_hdr, size_t req_hdr_size,
                  const void *req_payload, size_t req_payload_size,
                  struct nv_request *out) {
  struct scatterlist sg_req, sg_resp;
  struct scatterlist *sgs[2];
  void *req_buf = ndev->buf;
  void *resp_buf = (char *)ndev->buf + NV_BUF_SIZE / 2;
  size_t req_total = req_hdr_size + req_payload_size;
  size_t resp_max = NV_BUF_SIZE / 2;
  unsigned int len;
  void *token;
  int ret = 0;

  if (req_total > resp_max) {
    pr_err("nv_do_request: request too large (%zu bytes)\n", req_total);
    return -EINVAL;
  }

  mutex_lock(&ndev->vq_lock);

  /* Assemble the readable (request) buffer. */
  memcpy(req_buf, req_hdr, req_hdr_size);
  if (req_payload && req_payload_size)
    memcpy((char *)req_buf + req_hdr_size, req_payload, req_payload_size);

  /* Zero the writable (response) buffer so stale data can't leak. */
  memset(resp_buf, 0, resp_max);

  /* Build two scatter-gather entries. */
  sg_init_one(&sg_req, req_buf, req_total);
  sg_init_one(&sg_resp, resp_buf, resp_max);
  sgs[0] = &sg_req;
  sgs[1] = &sg_resp;

  /* Add the chain: 1 readable SG, 1 writable SG. */
  ret = virtqueue_add_sgs(ndev->vq, sgs, 1, /* num readable */
                          1,                /* num writable */
                          ndev /* token: anything non-NULL */, GFP_KERNEL);
  if (ret) {
    pr_err("nv_do_request: virtqueue_add_sgs failed: %d\n", ret);
    goto out_unlock;
  }

  /* Notify the backend. */
  virtqueue_kick(ndev->vq);

  /* Wait for the used-ring entry.
   * The condition is: virtqueue_get_buf() returns non-NULL. */
  wait_event(ndev->resp_wq, (token = virtqueue_get_buf(ndev->vq, &len)) != NULL);
  ret = 0;

  if (ret) {
    /* Interrupted by a signal.  The virtqueue entry is still
     * in-flight; for Phase 1 we just return the error and the
     * next operation will find the stale used entry.
     * Phase 3 will handle this properly with request cancellation. */
    pr_warn("nv_do_request: interrupted\n");
    goto out_unlock;
  }

  /* Parse the response header. */
  if (len < sizeof(struct resp_header)) {
    pr_err("nv_do_request: response too short: %u bytes\n", len);
    ret = -EIO;
    goto out_unlock;
  }

  memcpy(&out->resp_hdr, resp_buf, sizeof(struct resp_header));
  out->resp_payload_len = len - sizeof(struct resp_header);

  if (out->resp_payload_len > 0) {
    out->resp_payload = kmemdup((char *)resp_buf + sizeof(struct resp_header),
                                out->resp_payload_len, GFP_KERNEL);
    if (!out->resp_payload) {
      ret = -ENOMEM;
      goto out_unlock;
    }
  } else {
    out->resp_payload = NULL;
  }

out_unlock:
  mutex_unlock(&ndev->vq_lock);
  return ret;
}
