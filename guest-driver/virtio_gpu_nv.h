/* SPDX-License-Identifier: Apache-2.0 */
/*
 * virtio_gpu_nv.h — shared protocol header for virtio-gpu-nv
 *
 * This header is the C mirror of crates/protocol/src/messages.rs.
 * Every type defined here has an exact Rust counterpart; the sizes are
 * verified by the compile-time assertions in messages.rs.
 *
 * Descriptor chain layout (one chain per operation):
 *   [readable: msg_header + payload] → [writable: resp_header + payload]
 */

#ifndef VIRTIO_GPU_NV_H
#define VIRTIO_GPU_NV_H

#include <linux/types.h>

/* -------------------------------------------------------------------------
 * Virtio device and queue IDs
 * ---------------------------------------------------------------------- */

/* Use a vendor-specific device ID in the reserved range (0x8000–0xFFFF). */
#define VIRTIO_ID_GPU_NV 0x8042

/* We use a single request/response virtqueue for Phase 1. */
#define VQ_REQUEST 0
#define NUM_QUEUES 1

/* Maximum number of bytes in a single ioctl param buffer. */
#define NV_MAX_PARAM_SIZE 4096

/* -------------------------------------------------------------------------
 * Message type discriminants  (MsgHeader::msg_type)
 * ---------------------------------------------------------------------- */

#define NV_MSG_OPEN 1
#define NV_MSG_CLOSE 2
#define NV_MSG_IOCTL 3

/* -------------------------------------------------------------------------
 * Status codes  (resp_header::status)
 * ---------------------------------------------------------------------- */

#define NV_STATUS_OK 0
#define NV_STATUS_INVALID_MSG_TYPE 1
#define NV_STATUS_INVALID_DEVICE 2
#define NV_STATUS_OPEN_FAILED 3
#define NV_STATUS_BAD_HANDLE 4
#define NV_STATUS_IOCTL_FAILED 5
#define NV_STATUS_BUFFER_TOO_SMALL 6

/* -------------------------------------------------------------------------
 * Device kind constants  (open_req::kind)
 * ---------------------------------------------------------------------- */

#define NV_DEV_CTL 0 /* /dev/nvidiactl  */
#define NV_DEV_GPU 1 /* /dev/nvidia0..N */
#define NV_DEV_UVM 2 /* /dev/nvidia-uvm */

/* -------------------------------------------------------------------------
 * Common headers
 * ---------------------------------------------------------------------- */

/**
 * struct msg_header - Prefix of every request buffer.
 * @msg_type: One of NV_MSG_*.
 * @cookie:   Opaque value echoed back in resp_header::cookie.
 * @_pad:     Must be zero.
 *
 * sizeof = 16.
 */
struct msg_header {
  __le32 msg_type;
  __le32 _pad;
  __le64 cookie;
} __packed;

/**
 * struct resp_header - Prefix of every response buffer.
 * @status:     One of NV_STATUS_*.
 * @errno_host: Host errno when status indicates a syscall failure, else 0.
 * @cookie:     Echoed from msg_header::cookie.
 *
 * sizeof = 16.
 */
struct resp_header {
  __le32 status;
  __s32 errno_host;
  __le64 cookie;
} __packed;

/* -------------------------------------------------------------------------
 * OPEN  (NV_MSG_OPEN)
 * ---------------------------------------------------------------------- */

/**
 * struct open_req - Payload following msg_header for an OPEN request.
 * @kind:  NV_DEV_CTL / NV_DEV_GPU / NV_DEV_UVM.
 * @index: GPU index (0-based) when kind == NV_DEV_GPU, ignored otherwise.
 * @_pad:  Must be zero.
 *
 * sizeof = 8.
 */
struct open_req {
  __u8 kind;
  __u8 index;
  __u8 _pad[6];
} __packed;

/**
 * struct open_resp - Payload following resp_header for an OPEN response.
 * @guest_handle: Opaque handle for subsequent requests.  Valid only when
 *                resp_header::status == NV_STATUS_OK.
 *
 * sizeof = 8.
 */
struct open_resp {
  __le64 guest_handle;
} __packed;

/* -------------------------------------------------------------------------
 * CLOSE  (NV_MSG_CLOSE)
 * ---------------------------------------------------------------------- */

/**
 * struct close_req - Payload following msg_header for a CLOSE request.
 * @guest_handle: Handle returned by a previous open_resp.
 *
 * sizeof = 8.
 */
struct close_req {
  __le64 guest_handle;
} __packed;

/**
 * struct close_resp - Payload following resp_header for a CLOSE response.
 *
 * sizeof = 8.  Currently unused; resp_header::status carries all info.
 */
struct close_resp {
  __le64 _pad;
} __packed;

/* -------------------------------------------------------------------------
 * IOCTL  (NV_MSG_IOCTL)  — Phase 2
 * ---------------------------------------------------------------------- */

/**
 * struct ioctl_req - Payload following msg_header for an IOCTL request.
 * @guest_handle: Handle of the open file on which to issue the ioctl.
 * @request:      Linux ioctl number.
 * @param_size:   Number of raw parameter bytes that follow this struct.
 * @_pad:         Must be zero.
 *
 * sizeof = 24.  Raw parameter bytes follow immediately.
 */
struct ioctl_req {
  __le64 guest_handle;
  __le64 request;
  __le32 param_size;
  __le32 _pad;
} __packed;

/**
 * struct ioctl_resp - Payload following resp_header for an IOCTL response.
 * @param_size:  Number of raw parameter bytes that follow this struct.
 * @shm_offset: SHM BAR byte offset (non-zero for mapping ioctls only).
 * @shm_length: SHM region length in bytes (non-zero for mapping ioctls only).
 * @pgprot:     Page-protection hint: 0=WB, 1=WC, 2=UC.
 *
 * sizeof = 32.  Raw parameter bytes follow immediately.
 */
struct ioctl_resp {
  __le32 param_size;
  __le32 _pad;
  __le64 shm_offset;
  __le64 shm_length;
  __u8 pgprot;
  __u8 _pad2[7];
} __packed;

/* -------------------------------------------------------------------------
 * Per-mapping metadata (stored after a successful NV_ESC_RM_MAP_MEMORY)
 * ---------------------------------------------------------------------- */

/**
 * struct nv_mapping_info - Records SHM metadata from a mapping ioctl.
 * @list:       Linked into nv_file_ctx::mappings.
 * @shm_offset: SHM BAR byte offset (what userspace passes as mmap offset).
 * @shm_length: Length in bytes.
 * @pgprot:     Cache type: 0=WB, 1=WC, 2=UC.
 */
struct nv_mapping_info {
  struct list_head list;
  __u64 shm_offset;
  __u64 shm_length;
  __u8 pgprot;
};

/**
 * struct nv_file_ctx - State associated with one open() of /dev/nvidia*.
 * @guest_handle: Handle assigned by the backend on OPEN.
 * @dev:          Pointer to the owning nv_dev (for virtqueue access).
 * @mappings:     List of nv_mapping_info from successful mapping ioctls.
 * @mappings_lock: Protects the mappings list.
 */
struct nv_file_ctx {
  __u64 guest_handle;
  struct nv_dev *dev;
  struct list_head mappings;
  spinlock_t mappings_lock;
};

#endif /* VIRTIO_GPU_NV_H */
