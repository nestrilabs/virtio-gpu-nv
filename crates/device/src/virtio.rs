// crates/device/src/virtio.rs
//
// Public API surface for VMM integration.
//
// This module re-exports the types that a VMM (e.g. libkrun) needs to
// integrate virtio-gpu-nv.  The VMM implements the virtio transport and
// calls NvidiaBackend::dispatch() for each descriptor chain.
//
// See INTEGRATION.md for how to wire this into libkrun.

/// The virtio device ID that the guest driver probes for.
pub const VIRTIO_ID_GPU_NV: u32 = 0x8042;

/// Number of virtqueues (single request queue).
pub const NUM_QUEUES: usize = 1;

/// Recommended virtqueue size.
pub const QUEUE_SIZE: u16 = 256;

/// Device configuration space layout.
///
/// The VMM should expose this via virtio config reads.
/// The guest driver reads shm_bar_gpa to configure nv_mmap().
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct NvGpuConfig {
    pub num_gpus: u32,
    pub shm_bar_gpa: u64,
    pub shm_bar_size: u64,
}
