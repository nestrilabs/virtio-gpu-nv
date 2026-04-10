/* SPDX-License-Identifier: GPL-2.0 */
/*
 * nvgpu_rmalloc_classes.h — hClass → pAllocParms size lookup for RM_ALLOC
 *
 * When NVIDIA userspace calls RM_ALLOC with paramsSize == 0 but
 * pAllocParms != NULL, the host RM driver determines the allocation
 * parameter size from hClass internally.  The guest driver must know
 * how many bytes to copy_from_user before forwarding to the VMM.
 *
 * This table mirrors gVisor nvproxy's per-class alloc param type table
 * and the open-gpu-kernel-modules headers.
 *
 * Classes that use rmAllocNoParams (pAllocParms should be NULL) return 0.
 * Unknown classes get a conservative fallback.
 */

#ifndef NVGPU_RMALLOC_CLASSES_H
#define NVGPU_RMALLOC_CLASSES_H

#include <linux/types.h>

#define NVGPU_RMALLOC_FALLBACK_SIZE 512

static inline u32 nvgpu_rmalloc_class_param_size(u32 hClass) {
  switch (hClass) {

  /* ── Client / device / subdevice ── */
  case 0x0041:
    return 12; /* NV0000_ALLOC_PARAMETERS */
  case 0x0080:
    return 52; /* NV0080_ALLOC_PARAMETERS */
  case 0x2080:
    return 4; /* NV2080_ALLOC_PARAMETERS */
  case 0x2081:
    return 4; /* NV2081_ALLOC_PARAMETERS */

  /* ── Memory allocation ── */
  case 0x0002:
    return 64; /* NV_MEMORY_ALLOCATION_PARAMS (SYSTEM) */
  case 0x003e:
    return 64; /* NV_MEMORY_ALLOCATION_PARAMS (LOCAL_USER) */
  case 0x00fb:
    return 64; /* NV_MEMORY_ALLOCATION_PARAMS (VIRTUAL) */
  case 0x00fc:
    return 64; /* NV_MEMORY_ALLOCATION_PARAMS (NV01_MEMORY_VIRTUAL) */

  /* ── VA space ── */
  case 0x90f1:
    return 56; /* NV_VASPACE_ALLOCATION_PARAMETERS */

  /* ── Channel group ── */
  case 0xa06c:
    return 20; /* NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS */

  /* ── Channels (NV_CHANNEL_ALLOC_PARAMS, ~160 bytes) ── */
  case 0xb06f:
    return 160; /* TURING_CHANNEL_GPFIFO_A */
  case 0xc06f:
    return 160; /* AMPERE_CHANNEL_GPFIFO_A */
  case 0xc46f:
    return 160; /* HOPPER_CHANNEL_GPFIFO_A */

  /* ── Graphics / compute objects (NV_GR_ALLOCATION_PARAMETERS) ── */
  case 0xc597:
    return 8; /* TURING_A */
  case 0xc697:
    return 8; /* AMPERE_A */
  case 0xc797:
    return 8; /* ADA_A */
  case 0xcb97:
    return 8; /* HOPPER_A */
  case 0xc5c0:
    return 8; /* TURING_COMPUTE_A */
  case 0xc6c0:
    return 8; /* AMPERE_COMPUTE_A */
  case 0xc7c0:
    return 8; /* AMPERE_COMPUTE_B */
  case 0xc9c0:
    return 8; /* ADA_COMPUTE_A */
  case 0xcbc0:
    return 8; /* HOPPER_COMPUTE_A */

  /* ── 2D / inline-to-memory (NV_GR_ALLOCATION_PARAMETERS) ── */
  case 0x902d:
    return 8; /* FERMI_TWOD_A */
  case 0xa140:
    return 8; /* KEPLER_INLINE_TO_MEMORY_B */

  /* ── DMA copy (NVB0B5_ALLOCATION_PARAMETERS) ── */
  case 0xc5b5:
    return 4; /* TURING_DMA_COPY_A */
  case 0xc6b5:
    return 4; /* AMPERE_DMA_COPY_A */
  case 0xc7b5:
    return 4; /* AMPERE_DMA_COPY_B */
  case 0xcbb5:
    return 4; /* HOPPER_DMA_COPY_A */

  /* ── Video decode / encode (NV_BSP / NV_MSENC _ALLOCATION_PARAMETERS) ── */
  case 0xc4b0:
    return 8;
  case 0xc6b0:
    return 8;
  case 0xc7b0:
    return 8;
  case 0xc9b0:
    return 8;
  case 0xb8b0:
    return 8;
  case 0xc4b7:
    return 8;
  case 0xc7b7:
    return 8;
  case 0xc9b7:
    return 8;

  /* ── P2P / third-party P2P ── */
  case 0x503b:
    return 16; /* NV503B_ALLOC_PARAMETERS */
  case 0x503c:
    return 8; /* NV503C_ALLOC_PARAMETERS */

  /* ── Context share ── */
  case 0x9067:
    return 16; /* NV_CTXSHARE_ALLOCATION_PARAMETERS */

  /* ── Display ── */
  case 0x9072:
    return 8; /* NV9072_ALLOCATION_PARAMETERS */

  /* ── Event ── */
  case 0x0005:
    return 16; /* NV0005_ALLOC_PARAMETERS */

  /* ── Semaphore surface ── */
  case 0x00da:
    return 16; /* NV_SEMAPHORE_SURFACE_ALLOC_PARAMETERS */

  /* ── Memory fabric ── */
  case 0x00f8:
    return 32; /* NV00F8_ALLOCATION_PARAMETERS */
  case 0x00fd:
    return 32; /* NV00FD_ALLOCATION_PARAMETERS */

  /* ── Hopper usermode ── */
  case 0xc661:
    return 8; /* NV_HOPPER_USERMODE_A_PARAMS */

  /* ── Confidential compute ── */
  case 0xcb33:
    return 16; /* NV_CONFIDENTIAL_COMPUTE_ALLOC_PARAMS */

  /* ── Memory mapper ── */
  case 0x00fe:
    return 8; /* NV_MEMORY_MAPPER_ALLOCATION_PARAMS */

  /* ── SM debugger ── */
  case 0x83de:
    return 16; /* GT200_DEBUGGER alloc params */

  /* ── RM user shared data ── */
  case 0x00de:
    return 4; /* NV00DE_ALLOC_PARAMETERS */

  /* ── No-params classes (rmAllocNoParams) ── */
  case 0xc3b5:
    return 0; /* GF100_PROFILER */
  case 0xc570:
    return 0; /* TURING_USERMODE_A */
  case 0xc670:
    return 0; /* VOLTA_USERMODE_A */
  case 0xc4d1:
    return 0; /* HOPPER_SEC2_WORK_LAUNCH_A */
  case 0x0073:
    return 0; /* NV04_DISPLAY_COMMON */
  case 0x208f:
    return 0; /* NV20_SUBDEVICE_DIAG */
  case 0x9010:
    return 0; /* GF100_ZBC_CLEAR */
  case 0xa080:
    return 0; /* GF100_SUBDEVICE_MASTER */

  /* ── Unknown → conservative fallback ── */
  default:
    return NVGPU_RMALLOC_FALLBACK_SIZE;
  }
}

#endif /* NVGPU_RMALLOC_CLASSES_H */
