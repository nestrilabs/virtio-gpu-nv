// crates/abi/src/versions/v595_58_03.rs
//
// ABI table for NVIDIA driver version 595.58.03.

use crate::ioctl::*;

pub use super::v535_129_03::{IoctlEntry, IoctlKind};

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
        // IoctlNVOS33ParametersWithFD: NVOS33_PARAMETERS(48) + FD(4) + pad(4) = 56
        IoctlEntry {
            number: _IOWR(NV_ESC_RM_MAP_MEMORY, 56),
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
        // New in 580+: NV_ESC_RM_MAP_MEMORY_DMA.
        // NVOS46_PARAMETERS_V580 size = 48 bytes (from gVisor nvproxy v580_65_06).
        IoctlEntry {
            number: _IOWR(NV_ESC_RM_MAP_MEMORY_DMA, 48),
            escape: NV_ESC_RM_MAP_MEMORY_DMA,
            kind: IoctlKind::Simple,
        },
    ];
    TABLE
}
