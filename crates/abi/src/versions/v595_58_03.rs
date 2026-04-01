// crates/abi/src/versions/v595_58_03.rs
//
// ABI table for NVIDIA driver version 595.58.03.
//
// Derived from the 535.129.03 table and the gVisor nvproxy version chain
// (535 → 555 → 560 → 570 → 575 → 580 → 590 → 595).
//
// All frontend ioctl parameter struct sizes verified against the
// kernel-open headers (nvos.h, nv-ioctl.h) for 595.58.03:
//
//   NVOS00_PARAMETERS  (RM_FREE)          = 16 bytes  (unchanged)
//   NVOS64_PARAMETERS  (RM_ALLOC)         = 48 bytes  (unchanged)
//   NVOS54_PARAMETERS  (RM_CONTROL)       = 56 bytes  (unchanged)
//   NVOS33_PARAMETERS  (RM_MAP_MEMORY)    = 40 bytes  (unchanged)
//   NVOS34_PARAMETERS  (RM_UNMAP_MEMORY)  = 40 bytes  (unchanged)
//   nv_ioctl_register_fd_t                =  8 bytes  (unchanged, padded)
//   nv_ioctl_alloc_os_event_t             = 16 bytes  (unchanged)
//   nv_ioctl_free_os_event_t              = 16 bytes  (unchanged)
//   nv_ioctl_card_info_t * NV_MAX_GPUS    = 4096      (unchanged)
//   nv_ioctl_rm_api_version_t             = 4096*     (unchanged, xfer-style)
//
// The gVisor version chain shows no changes to frontendIoctl struct sizes
// from 535 through 595.  Changes in that range are limited to:
//   - New/modified controlCmd entries (nested RM_CONTROL dispatch)
//   - New/modified allocationClass entries (nested RM_ALLOC dispatch)
//   - New/modified uvmIoctl entries
//   - NV_ESC_RM_MAP_MEMORY_DMA added at 580 (NVOS46_PARAMETERS_V580)
//
// Phase 1 reuses the 535 table directly.  Phase 2 will add
// NV_ESC_RM_MAP_MEMORY_DMA and the full nested dispatch tables.

use crate::ioctl::*;

pub use super::v535_129_03::{IoctlEntry, IoctlKind};

/// Build the ioctl table for 595.58.03.
///
/// All frontend ioctl parameter struct sizes are identical to 535.129.03.
/// The table will diverge in Phase 2 when nested RM_CONTROL / RM_ALLOC
/// dispatch tables and the new NV_ESC_RM_MAP_MEMORY_DMA entry are added.
pub fn table() -> &'static [IoctlEntry] {
    static TABLE: &[IoctlEntry] = &[
        IoctlEntry {
            number: _IOWR(NV_ESC_CHECK_VERSION_STR, 4096),
            escape: NV_ESC_CHECK_VERSION_STR,
            kind: IoctlKind::Simple,
        },
        IoctlEntry {
            number: _IOWR(NV_ESC_CARD_INFO, 4096),
            escape: NV_ESC_CARD_INFO,
            kind: IoctlKind::Simple,
        },
        IoctlEntry {
            number: _IOWR(NV_ESC_REGISTER_FD, 8),
            escape: NV_ESC_REGISTER_FD,
            kind: IoctlKind::FdCarrying,
        },
        IoctlEntry {
            number: _IOWR(NV_ESC_RM_ALLOC, 48),
            escape: NV_ESC_RM_ALLOC,
            kind: IoctlKind::RmAlloc,
        },
        IoctlEntry {
            number: _IOWR(NV_ESC_RM_CONTROL, 56),
            escape: NV_ESC_RM_CONTROL,
            kind: IoctlKind::RmControl,
        },
        IoctlEntry {
            number: _IOWR(NV_ESC_RM_FREE, 16),
            escape: NV_ESC_RM_FREE,
            kind: IoctlKind::Simple,
        },
        IoctlEntry {
            number: _IOWR(NV_ESC_RM_MAP_MEMORY, 40),
            escape: NV_ESC_RM_MAP_MEMORY,
            kind: IoctlKind::Mapping,
        },
        IoctlEntry {
            number: _IOWR(NV_ESC_RM_UNMAP_MEMORY, 40),
            escape: NV_ESC_RM_UNMAP_MEMORY,
            kind: IoctlKind::Simple,
        },
        IoctlEntry {
            number: _IOWR(NV_ESC_ALLOC_OS_EVENT, 16),
            escape: NV_ESC_ALLOC_OS_EVENT,
            kind: IoctlKind::FdCarrying,
        },
        IoctlEntry {
            number: _IOWR(NV_ESC_FREE_OS_EVENT, 16),
            escape: NV_ESC_FREE_OS_EVENT,
            kind: IoctlKind::FdCarrying,
        },
    ];
    TABLE
}
