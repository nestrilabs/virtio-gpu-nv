// crates/abi/src/types.rs
//
// Common NVIDIA types shared across ioctl structs.
// Ported from gVisor pkg/abi/nvgpu/nvgpu.go.

/// Opaque RM object handle.  Clients allocate these; the driver tracks them.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NvHandle(pub u32);

impl NvHandle {
    pub const NULL: NvHandle = NvHandle(0);
}

/// NVIDIA status code returned inside ioctl parameter structs.
/// NV_OK == 0.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NvStatus(pub u32);

impl NvStatus {
    pub const OK: NvStatus = NvStatus(0);

    pub fn is_ok(self) -> bool {
        self.0 == 0
    }
}

/// 64-bit GPU virtual address (returned by mapping operations).
pub type NvP64   = u64;
/// GPU device memory address.
pub type NvU64   = u64;
pub type NvU32   = u32;
pub type NvU16   = u16;
pub type NvU8    = u8;
pub type NvBool  = u8;
pub type NvS32   = i32;
