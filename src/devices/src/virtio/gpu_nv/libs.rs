// src/devices/src/virtio/gpu_nv/libs.rs
//
// Locate the directory on the host that contains the NVIDIA userspace
// libraries (libcuda.so, libcudart.so, libnvidia-encode.so, …).
//
// The guest needs these libraries at the same version as the host kernel
// driver.  We share the host directory into the guest via virtio-fs so the
// guest always has a matching set without any manual installation.

use std::path::{Path, PathBuf};

/// Try to find the host NVIDIA library directory.
///
/// Tries `nvidia-container-cli` first (most reliable on modern setups),
/// then falls back to searching a list of common library paths.
///
/// Returns the directory path on success, or an error string if no NVIDIA
/// libraries can be found.
pub fn find_nvidia_libs() -> Result<PathBuf, String> {
    // ── Strategy 1: nvidia-container-cli list ────────────────────────────────
    if let Ok(output) = std::process::Command::new("nvidia-container-cli")
        .args(["list", "--libraries"])
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(first_line) = stdout.lines().next() {
                if let Some(parent) = Path::new(first_line.trim()).parent() {
                    if parent.exists() {
                        return Ok(parent.to_path_buf());
                    }
                }
            }
        }
    }

    // ── Strategy 2: search common library directories ────────────────────────
    let candidates = [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib",
        "/usr/local/lib",
    ];

    for dir in &candidates {
        // Presence of libcuda.so.1 is the canonical indicator.
        if Path::new(dir).join("libcuda.so.1").exists() {
            return Ok(PathBuf::from(dir));
        }
    }

    Err("Cannot find NVIDIA libraries on host. \
         Install the NVIDIA driver or ensure nvidia-container-cli is in PATH."
        .into())
}

/// Return a list of NVIDIA library names that must be shared with the guest.
///
/// Not every library in the directory is needed — this curated list covers
/// CUDA compute, Vulkan, OpenGL, NVENC, and NVDEC.
pub fn required_nvidia_libs() -> &'static [&'static str] {
    &[
        "libcuda.so.1",
        "libcuda.so",
        "libcudart.so",
        "libnvidia-ml.so.1",
        "libnvidia-ml.so",
        "libnvidia-encode.so.1",
        "libnvidia-encode.so",
        "libnvidia-decode.so.1",
        "libnvidia-decode.so",
        "libnvidia-opticalflow.so.1",
        "libnvidia-opencl.so.1",
        "libnvidia-opencl.so",
        "libOpenCL.so.1",
        "libnvidia-glcore.so",
        "libnvidia-eglcore.so",
        "libnvidia-glsi.so",
        "libnvidia-tls.so",
        "libnvidia-allocator.so.1",
        "libvdpau_nvidia.so",
        "nvidia_icd.json", // Vulkan ICD manifest
        "nvidia_layers.json",
    ]
}
