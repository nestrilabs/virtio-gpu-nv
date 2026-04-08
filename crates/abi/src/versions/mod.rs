// crates/abi/src/versions/mod.rs
//
// Per-version ABI tables.  Each sub-module defines the ioctl handler map
// for one NVIDIA driver version.  The backend selects the right table at
// runtime after `NV_ESC_CHECK_VERSION_STR` succeeds.

pub mod v535_129_03;
pub mod v595_58_03;

use crate::version::DriverVersion;
use std::cell::OnceCell;

pub const SUPPORTED: OnceCell<Vec<DriverVersion>> = OnceCell::new();

/// Returns true if the given version is supported.
pub fn is_supported(v: DriverVersion) -> bool {
    let cell = SUPPORTED;
    let supported = cell.get_or_init(|| {
        // Versions supported
        Vec::from([
            DriverVersion::new(535, 129, 3),
            DriverVersion::new(595, 58, 3),
        ])
    });
    supported.contains(&v)
}
