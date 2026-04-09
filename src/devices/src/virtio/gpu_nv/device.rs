// src/devices/src/virtio/gpu_nv/device.rs
//
// GpuNv — core state for one virtio-gpu-nv device instance.

use crate::virtio::gpu_nv::{NVGPU_CAP_COMPUTE, NVGPU_CAP_GRAPHICS, NVGPU_CAP_VIDEO};
use std::collections::HashMap;
use std::sync::Arc;

use crate::virtio::gpu_nv::allowlist::AllowedIoctls;

// ─────────────────────────────────────────────────────────────────────────────
// Generic host→guest proc file passthrough
// ─────────────────────────────────────────────────────────────────────────────

/// One host procfs file to be recreated verbatim in the guest.
#[derive(Debug, Clone)]
pub struct HostProcFile {
    /// Path relative to /proc — e.g. "driver/nvidia/gpus/0000:08:00.0/information"
    pub guest_path: String,
    /// Raw file content read from the host.
    pub content: String,
}

/// Recursively walk /proc/driver/nvidia/ and capture every readable file.
pub fn read_host_nvidia_proc_tree() -> Vec<HostProcFile> {
    let root = std::path::Path::new("/proc/driver/nvidia");
    let mut files = Vec::new();
    walk_proc_dir(root, &mut files);
    log::info!("virtio-gpu-nv: captured {} host proc files", files.len());
    files
}

fn walk_proc_dir(dir: &std::path::Path, out: &mut Vec<HostProcFile>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => { log::warn!("virtio-gpu-nv: cannot read {}: {}", dir.display(), e); return; }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let meta = match std::fs::metadata(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if meta.is_dir() {
            walk_proc_dir(&path, out);
            continue;
        }
        if !meta.is_file() { continue; }
        let relative = match path.strip_prefix("/proc/") {
            Ok(r) => r.to_string_lossy().to_string(),
            Err(_) => continue,
        };
        match std::fs::read_to_string(&path) {
            Ok(content) => {
                log::debug!("virtio-gpu-nv: captured {}", relative);
                out.push(HostProcFile { guest_path: relative, content });
            }
            Err(e) => log::debug!("virtio-gpu-nv: skipping {}: {}", relative, e),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU info — only what we need to identify the device; rest travels as raw text
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// PCI address string — used as the proc subdirectory name.
    pub pci_addr: String,
    /// Device minor number — used for /dev/nvidia<minor> matching.
    pub minor: u32,
    /// Raw content of /proc/driver/nvidia/gpus/<pci>/information on the host.
    pub information: HostProcFile,
}

fn parse_gpu_information(pci_addr: &str, text: &str) -> Option<GpuInfo> {
    let mut minor = None;

    for line in text.lines() {
        let parts: Vec<&str> = line.splitn(2, ':').collect();
        if parts.len() != 2 {
            continue;
        }
        if parts[0].trim() == "Device Minor" {
            minor = parts[1].trim().parse::<u32>().ok();
            break; // only field we need to parse
        }
    }

    let pci_addr = pci_addr.to_string();
    Some(GpuInfo {
        information: HostProcFile {
            guest_path: format!("driver/nvidia/gpus/{}/information", pci_addr),
            content: text.to_string(),
        },
        pci_addr,
        minor: minor?,
    })
}

pub fn read_host_gpu_info() -> Result<Vec<GpuInfo>, String> {
    let gpu_root = std::path::Path::new("/proc/driver/nvidia/gpus");

    let entries = std::fs::read_dir(gpu_root)
        .map_err(|e| format!("cannot read {}: {}", gpu_root.display(), e))?;

    let mut gpus: Vec<GpuInfo> = Vec::new();

    for entry in entries {
        let entry = entry.map_err(|e| format!("readdir error: {}", e))?;
        let pci_addr = entry
            .file_name()
            .into_string()
            .map_err(|_| "non-UTF8 PCI address directory".to_string())?;

        let info_path = entry.path().join("information");
        let text = std::fs::read_to_string(&info_path)
            .map_err(|e| format!("cannot read {}: {}", info_path.display(), e))?;

        match parse_gpu_information(&pci_addr, &text) {
            Some(gpu) => {
                log::info!(
                    "virtio-gpu-nv: found host GPU {} minor={}",
                    gpu.pci_addr,
                    gpu.minor
                );
                gpus.push(gpu);
            }
            None => log::warn!(
                "virtio-gpu-nv: skipping {}: missing Device Minor field",
                info_path.display()
            ),
        }
    }

    gpus.sort_by(|a, b| a.pci_addr.cmp(&b.pci_addr));

    if gpus.is_empty() {
        return Err("no NVIDIA GPUs found in /proc/driver/nvidia/gpus".to_string());
    }
    if gpus.len() > 8 {
        log::warn!("virtio-gpu-nv: found {} GPUs, clamping to 8", gpus.len());
        gpus.truncate(8);
    }

    Ok(gpus)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public configuration (set by krun_enable_nvidia before the VM starts)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GpuNvConfig {
    /// Number of GPUs exposed to this VM (1–8).
    pub num_gpus: u32,
    /// Capability bitmask (NVGPU_CAP_*).
    pub caps: u32,
    /// Host driver version string, e.g. "595.58.03".
    pub driver_version: String,
    /// One entry per exposed GPU, in wire order.
    pub gpus: Vec<GpuInfo>,
    /// Any additional host proc files to recreate verbatim in the guest.
    /// Extend this at the call site to pass through whatever.
    pub extra_proc: Vec<HostProcFile>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal structures
// ─────────────────────────────────────────────────────────────────────────────

/// A host-side file opened on behalf of a guest open().
pub(crate) struct HostFd {
    /// The actual host file descriptor.
    pub fd: std::fs::File,
    /// NVGPU_DEV_* constant.
    #[allow(unused)]
    pub device_type: u32,
    /// Mapping IDs associated with this FD — cleaned up on close.
    pub mapping_ids: Vec<u32>,
}

/// A GPU memory region exposed to the guest via a KVM memory slot.
pub(crate) struct GpuMapping {
    /// Host virtual address returned by mmap().
    pub host_ptr: *mut libc::c_void,
    /// Size of the mapping in bytes.
    pub size: u64,
    /// Guest physical address allocated by MmioAllocator.
    pub guest_phys_addr: u64,
    /// KVM memory slot index used for this mapping.
    pub kvm_slot: u32,
    /// Back-reference to the owning host FD handle.
    #[allow(unused)]
    pub host_fd_handle: u32,
}

// SAFETY: GpuMapping is only accessed from the single VMM worker thread.
unsafe impl Send for GpuMapping {}
unsafe impl Sync for GpuMapping {}

// ─────────────────────────────────────────────────────────────────────────────
// MMIO address allocator
// ─────────────────────────────────────────────────────────────────────────────

/// Bump allocator for guest physical addresses in the GPU MMIO window.
/// The window must be outside guest RAM so KVM does not confuse it with RAM.
pub(crate) struct MmioAllocator {
    #[allow(unused)]
    base: u64,
    next: u64,
    limit: u64,
}

impl MmioAllocator {
    /// Create an allocator over [base, base+size).
    pub fn new(base: u64, size: u64) -> Self {
        MmioAllocator {
            base,
            next: base,
            limit: base + size,
        }
    }

    /// Allocate a page-aligned region of `size` bytes.
    /// Returns the guest physical address, or None if exhausted.
    pub fn alloc(&mut self, size: u64) -> Option<u64> {
        let aligned = (self.next + 0xFFF) & !0xFFF;
        if aligned.checked_add(size)? > self.limit {
            return None;
        }
        self.next = aligned + size;
        Some(aligned)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GpuNv — the device
// ─────────────────────────────────────────────────────────────────────────────

pub struct GpuNv {
    /// Static configuration.
    pub(crate) config: GpuNvConfig,

    /// Guest FD handle → host FD state.
    pub(crate) fd_table: HashMap<u32, HostFd>,
    pub(crate) next_handle: u32,

    pub(crate) avail_features: u64,
    pub(crate) acked_features: u64,

    /// Mapping ID → KVM-backed GPU memory region.
    pub(crate) mappings: HashMap<u32, GpuMapping>,
    pub(crate) next_mapping_id: u32,

    /// Allocator for guest physical addresses used by GPU mappings.
    pub(crate) mmio_alloc: MmioAllocator,

    /// KVM VM fd — used to create / delete memory slots.
    pub(crate) vm_fd: Arc<kvm_ioctls::VmFd>,

    /// Next KVM memory slot index.
    pub(crate) next_kvm_slot: u32,

    /// Security allowlist — only vetted ioctl numbers are forwarded.
    pub(crate) allowed_ioctls: AllowedIoctls,

    // ── virtio bookkeeping (filled in during activate) ──
    pub(crate) queues: Vec<crate::virtio::DeviceQueue>,
    pub(crate) interrupt_transport: Option<crate::virtio::InterruptTransport>,
    pub(crate) guest_memory: Option<vm_memory::GuestMemoryMmap>,
}

impl GpuNv {
    /// Construct a new GpuNv device.
    ///
    /// `mmio_base` / `mmio_size` define the guest physical window used for GPU
    /// memory mappings.  A 4 GiB window (e.g. 0x20_0000_0000 .. 0x21_0000_0000)
    /// comfortably covers typical workloads.
    pub fn new(
        config: GpuNvConfig,
        vm_fd: Arc<kvm_ioctls::VmFd>,
        mmio_base: u64,
        mmio_size: u64,
        first_kvm_slot: u32,
    ) -> Self {
        use crate::virtio::gpu_nv::allowlist::AllowedIoctls;

        let caps = config.caps;
        // Build feature bits from config.caps
        let mut avail_features = 1u64 << 32; // VIRTIO_F_VERSION_1
        if config.caps & NVGPU_CAP_COMPUTE != 0 {
            avail_features |= 1 << 0; // F_UVM
        }
        if config.caps & NVGPU_CAP_VIDEO != 0 {
            avail_features |= 1 << 1; // F_ENCODE
        }
        if config.caps & NVGPU_CAP_GRAPHICS != 0 {
            avail_features |= 1 << 2; // F_GRAPHICS
        }

        GpuNv {
            config,
            fd_table: HashMap::new(),
            next_handle: 1,
            mappings: HashMap::new(),
            avail_features,
            acked_features: 0,
            next_mapping_id: 1,
            mmio_alloc: MmioAllocator::new(mmio_base, mmio_size),
            vm_fd,
            next_kvm_slot: first_kvm_slot,
            allowed_ioctls: AllowedIoctls::new(caps),
            queues: Vec::new(),
            interrupt_transport: None,
            guest_memory: None,
        }
    }

    pub fn id(&self) -> u32 {
        crate::virtio::gpu_nv::VIRTIO_ID_GPU_NV
    }
}
