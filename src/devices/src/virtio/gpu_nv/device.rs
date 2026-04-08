// src/devices/src/virtio/gpu_nv/device.rs
//
// GpuNv — core state for one virtio-gpu-nv device instance.

use crate::virtio::gpu_nv::{NVGPU_CAP_COMPUTE, NVGPU_CAP_GRAPHICS, NVGPU_CAP_VIDEO};
use std::collections::HashMap;

use crate::virtio::gpu_nv::allowlist::AllowedIoctls;

// ─────────────────────────────────────────────────────────────────────────────
// Public configuration (set by krun_enable_nvidia before the VM starts)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GpuNvConfig {
    /// Number of GPUs exposed to this VM (1–8).
    pub num_gpus: u32,
    /// Capability bitmask (NVGPU_CAP_*).
    pub caps: u32,
    /// Host driver version string, e.g. "535.129.03".
    pub driver_version: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal structures
// ─────────────────────────────────────────────────────────────────────────────

/// A host-side file opened on behalf of a guest open().
pub(crate) struct HostFd {
    /// The actual host file descriptor.
    pub fd: std::fs::File,
    /// NVGPU_DEV_* constant.
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
    pub(crate) vm_fd: std::sync::Arc<kvm_ioctls::VmFd>,

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
        vm_fd: std::sync::Arc<kvm_ioctls::VmFd>,
        mmio_base: u64,
        mmio_size: u64,
        first_kvm_slot: u32,
    ) -> Self {
        use crate::virtio::gpu_nv::allowlist::AllowedIoctls;

        let caps = config.caps;
        // Build feature bits from config.caps
        let mut avail_features = 0u64;
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
}
