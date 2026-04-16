/* SPDX-License-Identifier: GPL-2.0 */
/*
 * nvgpu_rm_intercepts.h — RM_CONTROL commands intercepted in the guest
 *
 * Some RM_CONTROL commands have nested params containing userspace pointers
 * that cannot be forwarded through the VMM.  When no working external V2
 * variant exists, the guest driver must handle them locally.
 *
 * Each intercepted command has a handler function that:
 *   - Reads the V1 nested params from guest userspace
 *   - Synthesises the response from locally-available data (config space,
 *     proc files, etc.) OR rewrites to a different command
 *   - Copies results back to the original guest userspace pointers
 *   - Sets NVOS54 status field
 *
 * Commands with working V2 external variants should go in
 * nvgpu_v1v2_rewrites.h instead.
 *
 * ─── Known multi-pointer RM_CONTROL commands ───
 *
 * 0x00000101  SYSTEM_GET_BUILD_VERSION     3 string ptrs    → intercept
 * 0x00000110  SYSTEM_GET_P2P_CAPS_V2       1 array ptr      → TODO
 * 0x20800288  GPU_GET_NVENC_SW_SESSION_INFO 1 array ptr     → TODO
 * 0x20800803  BIOS_GET_NBSI               1 data ptr        → TODO
 * 0x20801210  GR_GET_CTX_BUFFER_INFO       1 array ptr      → TODO
 * 0x20810107  VGPU_MGR_GET_PGPU_INFO       2 ptrs           → TODO (MIG)
 * 0x90960103  SWINTR_GET_INFO              1 array ptr       → TODO
 *
 * Most of the TODO commands are only used by advanced tools (MIG manager,
 * nvenc session queries, VBIOS extraction).  nvidia-smi + basic CUDA
 * only needs BUILD_VERSION.
 */

#ifndef NVGPU_RM_INTERCEPTS_H
#define NVGPU_RM_INTERCEPTS_H

#include <linux/minmax.h>
#include <linux/string.h>
#include <linux/types.h>
#include <linux/uaccess.h>

/* Forward declaration — defined in main driver source */
struct nvgpu_fd;

/* ─── Helper: write NV_OK (0) to NVOS54 status field (offset 28) ─── */

static inline int nvgpu_set_nvos54_status(void __user *uarg, u32 status) {
  __le32 val = cpu_to_le32(status);
  if (copy_to_user(uarg + 28, &val, sizeof(val)))
    return -EFAULT;
  return 0;
}

/* ─── 0x00000101: NV0000_CTRL_CMD_SYSTEM_GET_BUILD_VERSION ─────────
 *
 * V1 nested params (40 bytes):
 *   offset  0: sizeOfStrings            (u32)
 *   offset  4: pad                      (u32)
 *   offset  8: pDriverVersionBuffer     (u64) — userspace string ptr
 *   offset 16: pVersionBuffer           (u64) — userspace string ptr
 *   offset 24: pTitleBuffer             (u64) — userspace string ptr
 *   offset 32: changelistNumber         (u32)
 *   offset 36: officialChangelistNumber (u32)
 */
static inline long
nvgpu_intercept_get_build_version(struct nvgpu_fd *nfd, void __user *uarg,
                                  void __user *user_nested, u32 nested_size,
                                  const char *driver_version) {
  enum {
    V1_SIZE_OF_STRINGS = 0,
    V1_DRIVER_PTR = 8,
    V1_VERSION_PTR = 16,
    V1_TITLE_PTR = 24,
    V1_CHANGELIST = 32,
    V1_OFFICIAL_CL = 36,
    V1_TOTAL = 40,
  };
  static const char title[] = "NVIDIA UNIX Open Kernel Module";

  u8 v1[V1_TOTAL];
  u64 ptr_driver, ptr_version, ptr_title;
  u32 size_of_strings;
  u32 ver_len, title_len, max_len;

  if (!user_nested || nested_size < V1_TOTAL)
    return -EINVAL;

  if (copy_from_user(v1, user_nested, V1_TOTAL))
    return -EFAULT;

  memcpy(&size_of_strings, &v1[V1_SIZE_OF_STRINGS], sizeof(u32));
  memcpy(&ptr_driver, &v1[V1_DRIVER_PTR], sizeof(u64));
  memcpy(&ptr_version, &v1[V1_VERSION_PTR], sizeof(u64));
  memcpy(&ptr_title, &v1[V1_TITLE_PTR], sizeof(u64));

  ver_len = strlen(driver_version) + 1;
  title_len = sizeof(title); /* includes NUL */
  max_len = max3(ver_len, ver_len, title_len);

  /* Phase 1: all pointers NULL → return sizeOfStrings only */
  if (!ptr_driver && !ptr_version && !ptr_title) {
    memset(v1, 0, V1_TOTAL);
    memcpy(&v1[V1_SIZE_OF_STRINGS], &max_len, sizeof(u32));
    if (copy_to_user(user_nested, v1, V1_TOTAL))
      return -EFAULT;
    return nvgpu_set_nvos54_status(uarg, 0);
  }

  /* Phase 2: pointers non-NULL → validate and copy strings */
  if (size_of_strings < max_len)
    return -EINVAL;

  if (ptr_driver &&
      copy_to_user((void __user *)ptr_driver, driver_version, ver_len))
    return -EFAULT;

  if (ptr_version &&
      copy_to_user((void __user *)ptr_version, driver_version, ver_len))
    return -EFAULT;

  if (ptr_title && copy_to_user((void __user *)ptr_title, title, title_len))
    return -EFAULT;

  /* Update nested params: sizeOfStrings, zero changelists */
  memcpy(&v1[V1_SIZE_OF_STRINGS], &max_len, sizeof(u32));
  memset(&v1[V1_CHANGELIST], 0, 8);
  if (copy_to_user(user_nested, v1, V1_TOTAL))
    return -EFAULT;

  return nvgpu_set_nvos54_status(uarg, 0);
}

/* ─── 0x00003d05: NV0000_CTRL_CMD_OS_GET_CAPS ─────────────────────
 *
 * Platform-specific caps query. V1 nested params (16 bytes):
 *   offset  0: capsTblSize  (u32) — size of caps buffer
 *   offset  4: pad          (u32)
 *   offset  8: capsTbl      (u64) — userspace pointer to caps buffer
 *
 * The host RM can't dereference the guest's capsTbl pointer.
 * Return a zeroed caps table = "no special OS capabilities",
 * which is correct for a VM environment.
 */
static inline long nvgpu_intercept_os_get_caps(struct nvgpu_fd *nfd,
                                               void __user *uarg,
                                               void __user *user_nested,
                                               u32 nested_size) {
  u8 params[16];
  u32 caps_tbl_size;
  u64 caps_tbl_ptr;

  if (!user_nested || nested_size < 16)
    return -EINVAL;

  if (copy_from_user(params, user_nested, 16))
    return -EFAULT;

  memcpy(&caps_tbl_size, &params[0], sizeof(u32));
  memcpy(&caps_tbl_ptr, &params[8], sizeof(u64));

  /* If pointer is non-NULL, zero out the userspace caps buffer */
  if (caps_tbl_ptr && caps_tbl_size > 0) {
    if (clear_user((void __user *)caps_tbl_ptr, caps_tbl_size))
      return -EFAULT;
  }

  return nvgpu_set_nvos54_status(uarg, 0); /* NV_OK */
}

/* ─── Dispatch table ──────────────────────────────────────────────── */

/*
 * nvgpu_try_intercept_rm_control — check if an RM_CONTROL cmd must be
 * handled locally.  Returns true if intercepted (ret is set), false
 * if the caller should proceed with normal VMM forwarding.
 */
static inline bool
nvgpu_try_intercept_rm_control(struct nvgpu_fd *nfd, u32 ctl_cmd,
                               void __user *uarg, void __user *user_nested,
                               u32 nested_size, const char *driver_version,
                               long *ret) {
  switch (ctl_cmd) {
  case 0x00000101: /* SYSTEM_GET_BUILD_VERSION */
    *ret = nvgpu_intercept_get_build_version(nfd, uarg, user_nested,
                                             nested_size, driver_version);
    return true;

    /*
     * TODO: Add future intercepts here as needed:
     *
     * case 0x00000110: SYSTEM_GET_P2P_CAPS_V2
     * case 0x20800288: GPU_GET_NVENC_SW_SESSION_INFO
     * case 0x20800803: BIOS_GET_NBSI
     * case 0x20801210: GR_GET_CTX_BUFFER_INFO
     * case 0x90960103: SWINTR_GET_INFO
     */

  case 0x00003d05: /* NV0000_CTRL_CMD_OS_GET_CAPS */
    *ret = nvgpu_intercept_os_get_caps(nfd, uarg, user_nested, nested_size);
    return true;

  default:
    return false;
  }
}

#endif /* NVGPU_RM_INTERCEPTS_H */
