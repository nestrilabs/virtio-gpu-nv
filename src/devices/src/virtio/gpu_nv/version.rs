// src/devices/src/virtio/gpu_nv/version.rs
//
// Detect the version of the NVIDIA kernel driver installed on the host.
// Called once at VMM startup; the result is placed in virtio config space
// so the guest driver can emit the correct version string.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriverVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl DriverVersion {
    /// Read /proc/driver/nvidia/version on the host and parse the version
    /// triple.  Returns an error if the file is absent (driver not loaded)
    /// or the version cannot be parsed.
    ///
    /// Example line:
    ///   NVRM version: NVIDIA UNIX x86_64 Kernel Module  535.129.03  ...
    pub fn detect() -> Result<Self, String> {
        let content = std::fs::read_to_string("/proc/driver/nvidia/version")
            .map_err(|e| format!("NVIDIA driver not loaded: {}", e))?;

        for word in content.split_whitespace() {
            let parts: Vec<&str> = word.split('.').collect();
            if parts.len() == 3 {
                if let (Ok(maj), Ok(min), Ok(pat)) = (
                    parts[0].parse::<u32>(),
                    parts[1].parse::<u32>(),
                    parts[2].parse::<u32>(),
                ) {
                    // Sanity-check: real NVIDIA driver versions start at 5xx.
                    if maj >= 500 {
                        return Ok(DriverVersion {
                            major: maj,
                            minor: min,
                            patch: pat,
                        });
                    }
                }
            }
        }

        Err("Could not parse NVIDIA driver version from /proc/driver/nvidia/version".into())
    }

    /// Format as the canonical "MMM.mm.pp" string used in config space and
    /// in /proc/driver/nvidia/version inside the guest.
    pub fn as_string(&self) -> String {
        format!("{}.{:02}.{:02}", self.major, self.minor, self.patch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_version_string() {
        // Simulate a /proc/driver/nvidia/version line.
        let content =
            "NVRM version: NVIDIA UNIX x86_64 Kernel Module  535.129.03  Fri Oct 27 05:09:12 UTC 2023\n\
             GCC version: gcc version 12.2.0";

        // Manually run the same parsing logic.
        let mut found: Option<DriverVersion> = None;
        for word in content.split_whitespace() {
            let parts: Vec<&str> = word.split('.').collect();
            if parts.len() == 3 {
                if let (Ok(maj), Ok(min), Ok(pat)) = (
                    parts[0].parse::<u32>(),
                    parts[1].parse::<u32>(),
                    parts[2].parse::<u32>(),
                ) {
                    if maj >= 500 {
                        found = Some(DriverVersion {
                            major: maj,
                            minor: min,
                            patch: pat,
                        });
                        break;
                    }
                }
            }
        }

        let v = found.expect("should have found version");
        assert_eq!(v.major, 535);
        assert_eq!(v.minor, 129);
        assert_eq!(v.patch, 3);
        assert_eq!(v.as_string(), "535.129.03");
    }
}
