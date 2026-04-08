// src/devices/src/virtio/gpu_nv/mod.rs
//
// virtio-gpu-nv VMM backend — module root.
//
// Compile only when the `nvidia` cargo feature is enabled:
//   cargo build --release --features nvidia

pub mod allowlist;
pub mod device;
pub mod handler;
pub mod libs;
pub mod mmap;
pub mod version;
pub mod virtio;

pub use device::GpuNv;
pub use device::GpuNvConfig;

// ─────────────────────────────────────────────────────────────────────────────
// Wire-protocol constants (shared between host-side modules and tests).
// ─────────────────────────────────────────────────────────────────────────────

pub const VIRTIO_ID_GPU_NV: u32 = 45;

/// Message types on the control virtqueue.
pub const NVGPU_MSG_OPEN: u32 = 1;
pub const NVGPU_MSG_CLOSE: u32 = 2;
pub const NVGPU_MSG_IOCTL: u32 = 3;
pub const NVGPU_MSG_MMAP: u32 = 4;
pub const NVGPU_MSG_MUNMAP: u32 = 5;

/// device_type values for OPEN.
pub const NVGPU_DEV_CTL: u32 = 255;
pub const NVGPU_DEV_UVM: u32 = 256;
pub const NVGPU_DEV_UVM_TOOLS: u32 = 257;
pub const NVGPU_DEV_MODESET: u32 = 258;

/// Capability bitmask values (also written to virtio config space).
pub const NVGPU_CAP_COMPUTE: u32 = 1 << 0;
pub const NVGPU_CAP_GRAPHICS: u32 = 1 << 1;
pub const NVGPU_CAP_VIDEO: u32 = 1 << 2;
pub const NVGPU_CAP_UTILITY: u32 = 1 << 3;

// ─────────────────────────────────────────────────────────────────────────────
// Wire-protocol structs (repr(C, packed) so they match the guest definitions).
// ─────────────────────────────────────────────────────────────────────────────

/// Common header for every message.
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NvgpuMsgHdr {
    pub msg_type: u32,
    pub handle: u32,
    pub status: i32,
    pub padding: u32,
}

/// OPEN request.
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NvgpuOpenReq {
    pub hdr: NvgpuMsgHdr,
    pub device_type: u32,
    pub flags: u32,
}

/// OPEN response (handle is in hdr.handle on success).
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NvgpuOpenResp {
    pub hdr: NvgpuMsgHdr,
}

/// IOCTL request header (followed by data_len bytes, then nested_len bytes).
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NvgpuIoctlReq {
    pub hdr: NvgpuMsgHdr,
    pub cmd: u32,
    pub data_len: u32,
    pub nested_offset: u32,
    pub nested_len: u32,
}

/// IOCTL response header (followed by data_len bytes, then nested_len bytes).
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NvgpuIoctlResp {
    pub hdr: NvgpuMsgHdr,
    pub data_len: u32,
    pub nested_len: u32,
}

/// MMAP request.
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NvgpuMmapReq {
    pub hdr: NvgpuMsgHdr,
    pub size: u64,
    pub offset: u64,
    pub prot: u32,
    pub padding: u32,
}

/// MMAP response.
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NvgpuMmapResp {
    pub hdr: NvgpuMsgHdr,
    pub guest_phys_addr: u64,
    pub size: u64,
    pub mapping_id: u32,
    pub padding: u32,
}

/// MUNMAP request.
#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NvgpuMunmapReq {
    pub hdr: NvgpuMsgHdr,
    pub mapping_id: u32,
    pub padding: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper — reinterpret a packed struct as a byte slice.
// ─────────────────────────────────────────────────────────────────────────────

pub fn bytes_of<T: Copy>(val: &T) -> Vec<u8> {
    let size = std::mem::size_of::<T>();
    let mut buf = vec![0u8; size];
    unsafe {
        std::ptr::copy_nonoverlapping(val as *const T as *const u8, buf.as_mut_ptr(), size);
    }
    buf
}

/// Extract the IOC_NR byte from a Linux ioctl command word.
pub fn ioc_nr(cmd: u32) -> u32 {
    cmd & 0xFF
}
