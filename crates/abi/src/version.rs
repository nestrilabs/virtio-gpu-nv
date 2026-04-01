// crates/abi/src/version.rs
//
// NVIDIA driver version representation and parsing.
//
// Ported from gVisor pkg/sentry/devices/nvproxy/version.go.

use core::fmt;

/// A parsed NVIDIA driver version, e.g. `535.129.03`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DriverVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl DriverVersion {
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch }
    }

    /// Parse from the string returned by `NV_ESC_CHECK_VERSION_STR`.
    /// Expected format: `"535.129.03"`.
    pub fn parse(s: &str) -> Option<Self> {
        let mut parts = s.trim().splitn(3, '.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        let patch = parts.next()?.parse().ok()?;
        Some(Self { major, minor, patch })
    }
}

impl fmt::Display for DriverVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{:02}", self.major, self.minor, self.patch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let v = DriverVersion::new(535, 129, 3);
        assert_eq!(v.to_string(), "535.129.03");
    }

    #[test]
    fn parse_ok() {
        assert_eq!(
            DriverVersion::parse("535.129.03"),
            Some(DriverVersion::new(535, 129, 3))
        );
    }
}
