// crates/abi/src/versions/v535_129_03.rs
//
// ABI table for NVIDIA driver version 535.129.03.
//
// Ported from gVisor pkg/sentry/devices/nvproxy/version.go,
// function `v535_129_03()`.
//
// Phase 1 records only the ioctl categories needed for open/close.
// Phase 2 will fill in the per-ioctl handler kinds.

use crate::ioctl::*;

/// The category of ioctl handling the backend must apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoctlKind {
    /// Copy params in, call host ioctl, copy params out.  No handles, no mmaps.
    Simple,
    /// Contains an embedded file descriptor that must be translated
    /// from guest handle to host fd before the host ioctl is issued.
    FdCarrying,
    /// Produces a new host mmap region; backend allocates SHM and returns offset.
    Mapping,
    /// `NV_ESC_RM_CONTROL`: nested dispatch on the embedded control command.
    RmControl,
    /// `NV_ESC_RM_ALLOC`: nested dispatch on the embedded allocation class.
    RmAlloc,
}

/// A single entry in the per-version ioctl table.
pub struct IoctlEntry {
    /// Linux ioctl number (as returned by `_IOWR(NV_ESC_*, size)`).
    pub number: u64,
    /// Raw NV_ESC_* index (for logging / debugging).
    pub escape: u32,
    pub kind: IoctlKind,
}

/// Build the ioctl table for 535.129.03.
///
/// Returns a static slice; the backend searches it by `number` at dispatch time.
/// The sizes below are for this specific driver version and must be updated
/// for each new version.  Sizes were obtained from nvproxy's Go structs.
pub fn table() -> &'static [IoctlEntry] {
    // NOTE: sizes are `sizeof` the C parameter struct for this driver version.
    // Phase 2 will add the full set; here we list the ones needed for Phase 1
    // testing (open/close don't go through this table, but we include
    // CHECK_VERSION_STR so Phase 2 tests work immediately).
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
    ];
    TABLE
}
