// crates/abi/src/ioctl.rs
//
// NV_ESC_* ioctl number constants and helpers.
//
// Source: NVIDIA open-gpu-kernel-modules
//   kernel-open/common/inc/nv-ioctl-numbers.h
//   kernel-open/common/inc/nv-ioctl.h

// ---------------------------------------------------------------------------
// /dev/nvidiactl and /dev/nvidia# ioctl numbers
// ---------------------------------------------------------------------------

pub const NV_ESC_CARD_INFO:              u32 = 0x01;
pub const NV_ESC_CHECK_VERSION_STR:      u32 = 0x25;
pub const NV_ESC_REGISTER_FD:            u32 = 0x2A;
pub const NV_ESC_ALLOC_OS_EVENT:         u32 = 0x2C;
pub const NV_ESC_FREE_OS_EVENT:          u32 = 0x2D;
pub const NV_ESC_STATUS_CODE:            u32 = 0x30;
pub const NV_ESC_RM_ALLOC_MEMORY:        u32 = 0x52;
pub const NV_ESC_RM_FREE:                u32 = 0x53;
pub const NV_ESC_RM_CONTROL:             u32 = 0x54;
pub const NV_ESC_RM_ALLOC:              u32 = 0x55;
pub const NV_ESC_RM_DUP_OBJECT:         u32 = 0x56;
pub const NV_ESC_RM_SHARE:              u32 = 0x57;
pub const NV_ESC_RM_MAP_MEMORY:         u32 = 0x4E;
pub const NV_ESC_RM_UNMAP_MEMORY:       u32 = 0x4F;
pub const NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO: u32 = 0x5E;

// ---------------------------------------------------------------------------
// /dev/nvidia-uvm ioctl numbers  (Phase 4)
// ---------------------------------------------------------------------------

pub const UVM_INITIALIZE:               u32 = 0x01;
pub const UVM_DEINITIALIZE:             u32 = 0x02;

// ---------------------------------------------------------------------------
// Linux ioctl number encoding helpers
// ---------------------------------------------------------------------------
//
// Linux encodes ioctl numbers as:
//   bits 31-30: direction (00=none, 01=write, 10=read, 11=read+write)
//   bits 29-16: size of argument (14 bits)
//   bits 15- 8: type (magic number)
//   bits  7- 0: number
//
// NVIDIA uses magic 'F' (0x46) for /dev/nvidia* ioctls.

const NV_IOCTL_MAGIC: u32 = b'F' as u32;

pub const fn _IOC(dir: u32, ty: u32, nr: u32, size: u32) -> u64 {
    ((dir << 30) | (size << 16) | (ty << 8) | nr) as u64
}

pub const fn _IO(nr: u32) -> u64 {
    _IOC(0, NV_IOCTL_MAGIC, nr, 0)
}

pub const fn _IOW(nr: u32, size: u32) -> u64 {
    _IOC(1, NV_IOCTL_MAGIC, nr, size)
}

pub const fn _IOR(nr: u32, size: u32) -> u64 {
    _IOC(2, NV_IOCTL_MAGIC, nr, size)
}

pub const fn _IOWR(nr: u32, size: u32) -> u64 {
    _IOC(3, NV_IOCTL_MAGIC, nr, size)
}
