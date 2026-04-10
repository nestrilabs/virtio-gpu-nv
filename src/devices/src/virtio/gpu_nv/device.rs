// src/devices/src/virtio/gpu_nv/device.rs
//
// GpuNv — core state for one virtio-gpu-nv device instance.

use crate::virtio::gpu_nv::{NVGPU_CAP_COMPUTE, NVGPU_CAP_GRAPHICS, NVGPU_CAP_VIDEO};
use std::collections::HashMap;
use std::sync::Arc;

use crate::virtio::gpu_nv::allowlist::AllowedIoctls;

/// One host sysfs file to be recreated verbatim in the guest.
#[derive(Debug, Clone)]
pub struct SysFile {
    /// Path relative to /sys/ — e.g. "devices/system/node/node0/cpumap"
    pub path: String,
    /// Raw file content.
    pub content: Vec<u8>,
}

/// A DRI device node on the host that belongs to one of our passed-through GPUs.
#[derive(Debug, Clone)]
pub struct DriDevice {
    /// Device node name — "renderD128" or "card1" etc.
    pub name: String,
    pub major: u32,
    pub minor: u32,
    /// NVIDIA GPU ID
    pub gpu_id: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Host /sys/module/nvidia* passthrough
// ─────────────────────────────────────────────────────────────────────────────

const SYS_MODULE_PATHS: &[&str] = &[
    "/sys/module/nvidia/initstate",
    "/sys/module/nvidia_uvm/initstate",
];

pub fn read_host_module_sys_files() -> Vec<SysFile> {
    let mut files = Vec::new();
    for &abs_path in SYS_MODULE_PATHS {
        match std::fs::read(abs_path) {
            Ok(content) => {
                let rel = abs_path.strip_prefix("/sys/").unwrap_or(abs_path);
                log::debug!("virtio-gpu-nv: captured sys {}", rel);
                files.push(SysFile {
                    path: rel.to_string(),
                    content,
                });
            }
            Err(e) => log::debug!("virtio-gpu-nv: skipping {}: {}", abs_path, e),
        }
    }
    log::info!("virtio-gpu-nv: captured {} module sys files", files.len());
    files
}

// ─────────────────────────────────────────────────────────────────────────────
// Host /sys/bus/pci/devices/<addr>/ passthrough
// ─────────────────────────────────────────────────────────────────────────────

fn walk_sysfs_tree(dir: &std::path::Path, prefix: &str, files: &mut Vec<SysFile>, max_size: u64) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();

        // Skip binary MMIO resource windows (resource0, resource1, …)
        // but keep the text "resource" file that lists all windows.
        if name.starts_with("resource") && name != "resource" {
            continue;
        }
        // Skip VBIOS rom (can be large)
        if name == "rom" {
            continue;
        }

        let meta = match std::fs::symlink_metadata(&path) {
            Ok(m) => m,
            Err(_) => continue,
        };

        // Skip symlinks (driver, subsystem, iommu, etc.)
        if meta.is_symlink() {
            continue;
        }

        if meta.is_dir() {
            // Skip directories that add no value for NVML / vulkan
            if name == "power"
                || name == "msi_irqs"
                || name.starts_with("virtfn")
                || name == "iommu"
                || name == "ptm"
                || name == "accel"
            {
                continue;
            }
            let sub_prefix = format!("{}/{}", prefix, name);
            walk_sysfs_tree(&path, &sub_prefix, files, max_size);
            continue;
        }

        if !meta.is_file() {
            continue;
        }

        if meta.len() > max_size {
            continue;
        }

        match std::fs::read(&path) {
            Ok(content) => {
                let rel = format!("{}/{}", prefix, name);
                log::debug!("virtio-gpu-nv: captured PCI sys {}", rel);
                files.push(SysFile { path: rel, content });
            }
            Err(e) => log::debug!("virtio-gpu-nv: skipping {}: {}", path.display(), e),
        }
    }
}

/// Read all readable sysfs files under /sys/bus/pci/devices/<addr>/
/// for each passed-through GPU.
pub fn read_host_pci_sysfs(gpu_pci_addrs: &[String]) -> Vec<SysFile> {
    let mut files = Vec::new();
    for addr in gpu_pci_addrs {
        let pci_dir = std::path::Path::new("/sys/bus/pci/devices").join(addr);
        if !pci_dir.exists() {
            log::debug!("virtio-gpu-nv: PCI dir {} not found", pci_dir.display());
            continue;
        }
        let prefix = format!("bus/pci/devices/{}", addr);
        walk_sysfs_tree(&pci_dir, &prefix, &mut files, 4096);
    }
    log::info!("virtio-gpu-nv: captured {} PCI sys files", files.len());
    files
}

pub fn read_host_sys_files(gpu_pci_addrs: &[String]) -> Vec<SysFile> {
    let mut files = Vec::new();

    // ── PCI sysfs files — one set per GPU ────────────────────────────────────
    for pci_addr in gpu_pci_addrs {
        let pci_base = std::path::Path::new("/sys/bus/pci/devices").join(pci_addr);

        match std::fs::read_dir(&pci_base) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let path = entry.path();
                    let meta = match std::fs::metadata(&path) {
                        Ok(m) => m,
                        Err(_) => continue,
                    };
                    if !meta.is_file() {
                        continue;
                    }
                    match std::fs::read(&path) {
                        Ok(content) => {
                            let rel = path
                                .strip_prefix("/sys/")
                                .unwrap_or(&path)
                                .to_string_lossy()
                                .to_string();
                            log::debug!("virtio-gpu-nv: captured pci sysfs {}", rel);
                            files.push(SysFile { path: rel, content });
                        }
                        Err(e) => log::debug!("virtio-gpu-nv: skipping {}: {}", path.display(), e),
                    }
                }
            }
            Err(e) => log::warn!("virtio-gpu-nv: cannot read {}: {}", pci_base.display(), e),
        }
    }

    log::info!("virtio-gpu-nv: captured {} host sys files", files.len());
    files
}

/// Parse "GPU ID: 0x1234abcd" from an nvidia information file.
/// Returns 0 on failure — callers should treat 0 as unknown, not fatal.
fn parse_gpu_id_from_information(text: &str) -> u32 {
    for line in text.lines() {
        // Line looks like:  "GPU ID:                  0x12345678"
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("GPU ID:") {
            let val = rest.trim();
            // May be hex ("0x...") or decimal
            if let Some(hex) = val.strip_prefix("0x").or_else(|| val.strip_prefix("0X")) {
                if let Ok(v) = u32::from_str_radix(hex, 16) {
                    return v;
                }
            } else if let Ok(v) = val.parse::<u32>() {
                return v;
            }
        }
    }
    0
}

/// Find DRI device nodes that belong to `gpu_pci_addrs`.
///
/// For each entry in /sys/class/drm/ the `device` symlink is resolved and
/// the last path component (the PCI address) is compared against our list.
/// The device major:minor is read from the adjacent `dev` file.
/// The gpu_id is read from /proc/driver/nvidia/gpus/<pci_addr>/information.
pub fn find_dri_devices(gpu_pci_addrs: &[String]) -> Vec<DriDevice> {
    let drm_class = std::path::Path::new("/sys/class/drm");
    let mut devices = Vec::new();

    let entries = match std::fs::read_dir(drm_class) {
        Ok(e) => e,
        Err(e) => {
            log::warn!("virtio-gpu-nv: cannot read /sys/class/drm: {}", e);
            return devices;
        }
    };

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();

        // Only card* and renderD* — not connectors / ports.
        if !name.starts_with("card") && !name.starts_with("renderD") {
            continue;
        }

        // Resolve /sys/class/drm/<name>/device → real PCI device path.
        let device_link = entry.path().join("device");
        let target = match std::fs::read_link(&device_link) {
            Ok(t) => t,
            Err(_) => continue,
        };

        let pci_addr = match target.file_name() {
            Some(n) => n.to_string_lossy().to_string(),
            None => continue,
        };

        if !gpu_pci_addrs.iter().any(|a| *a == pci_addr) {
            continue;
        }

        // /sys/class/drm/<name>/dev contains "major:minor\n"
        let dev_file = entry.path().join("dev");
        let dev_str = match std::fs::read_to_string(&dev_file) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let parts: Vec<&str> = dev_str.trim().split(':').collect();
        if parts.len() != 2 {
            continue;
        }

        let major: u32 = parts[0].parse().unwrap_or(0);
        let minor: u32 = parts[1].parse().unwrap_or(0);
        if major == 0 {
            continue;
        }

        // Read gpu_id from /proc/driver/nvidia/gpus/<pci_addr>/information.
        // Non-fatal — gpu_id=0 means the stub will return 0 for GET_DEV_INFO,
        // which is still better than failing the open.
        let gpu_id = {
            let info_path = format!("/proc/driver/nvidia/gpus/{}/information", pci_addr);
            match std::fs::read_to_string(&info_path) {
                Ok(text) => parse_gpu_id_from_information(&text),
                Err(e) => {
                    log::warn!(
                        "virtio-gpu-nv: cannot read {}: {} — gpu_id will be 0",
                        info_path,
                        e
                    );
                    0
                }
            }
        };

        log::info!(
            "virtio-gpu-nv: found DRI device {} ({}:{}) gpu_id=0x{:x} for GPU {}",
            name,
            major,
            minor,
            gpu_id,
            pci_addr
        );
        devices.push(DriDevice {
            name,
            major,
            minor,
            gpu_id,
        });
    }

    // Stable order: card* before renderD*, then by minor.
    devices.sort_by(|a, b| {
        let ak = if a.name.starts_with("card") { 0 } else { 1 };
        let bk = if b.name.starts_with("card") { 0 } else { 1 };
        ak.cmp(&bk).then(a.minor.cmp(&b.minor))
    });

    devices
}

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
        Err(e) => {
            log::warn!("virtio-gpu-nv: cannot read {}: {}", dir.display(), e);
            return;
        }
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
        if !meta.is_file() {
            continue;
        }
        let relative = match path.strip_prefix("/proc/") {
            Ok(r) => r.to_string_lossy().to_string(),
            Err(_) => continue,
        };
        match std::fs::read_to_string(&path) {
            Ok(content) => {
                log::debug!("virtio-gpu-nv: captured {}", relative);
                out.push(HostProcFile {
                    guest_path: relative,
                    content,
                });
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
    /// Host sysfs files to recreate in the guest (NUMA node, etc.).
    pub sys_files: Vec<SysFile>,
    /// DRI device nodes that belong to the passed-through GPU(s).
    pub dri_devices: Vec<DriDevice>,
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
    #[allow(unused)]
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
