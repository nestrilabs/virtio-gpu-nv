// crates/abi/src/versions/mod.rs
//
// Per-version ABI tables.  Each sub-module defines the ioctl handler map
// for one NVIDIA driver version.  The backend selects the right table at
// runtime after `NV_ESC_CHECK_VERSION_STR` succeeds.

pub mod v535_129_03;

use crate::version::DriverVersion;

/// The single version supported in Phase 1/2.
pub const SUPPORTED: DriverVersion = DriverVersion::new(535, 129, 3);

/// Returns true if the given version is supported.
pub fn is_supported(v: DriverVersion) -> bool {
    // For now, exact match only.  Relax to a range later.
    v == SUPPORTED
}
