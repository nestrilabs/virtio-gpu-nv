// crates/abi/src/ioctl.rs
//
// NV_ESC_* ioctl number constants and helpers.
//
// Two separate escape namespaces exist:
//
// 1. Frontend escapes (nv-ioctl-numbers.h) — NV_IOCTL_BASE (200) + offset.
//    Used for /dev/nvidiactl and /dev/nvidia# node-level operations.
//
// 2. RM escapes (nv_escape.h) — small numbers (0x27–0x5F).
//    Used for Resource Manager operations, also dispatched via the same
//    /dev/nvidia* ioctl path.
//
// Both are encoded into the _IOC_NR field (bits 7:0) of the Linux ioctl
// number.  They never collide because frontend escapes are ≥200 and RM
// escapes are ≤0x5F (95).

#![allow(non_snake_case)]

// ---------------------------------------------------------------------------
// Frontend escapes — from nv-ioctl-numbers.h
// ---------------------------------------------------------------------------

const NV_IOCTL_BASE: u32 = 200;

pub const NV_ESC_CARD_INFO: u32 = NV_IOCTL_BASE + 0; // 200 = 0xC8
pub const NV_ESC_REGISTER_FD: u32 = NV_IOCTL_BASE + 1; // 201 = 0xC9
pub const NV_ESC_ALLOC_OS_EVENT: u32 = NV_IOCTL_BASE + 6; // 206 = 0xCE
pub const NV_ESC_FREE_OS_EVENT: u32 = NV_IOCTL_BASE + 7; // 207 = 0xCF
pub const NV_ESC_STATUS_CODE: u32 = NV_IOCTL_BASE + 9; // 209 = 0xD1
pub const NV_ESC_CHECK_VERSION_STR: u32 = NV_IOCTL_BASE + 10; // 210 = 0xD2
pub const NV_ESC_IOCTL_XFER_CMD: u32 = NV_IOCTL_BASE + 11; // 211 = 0xD3
pub const NV_ESC_ATTACH_GPUS_TO_FD: u32 = NV_IOCTL_BASE + 12; // 212 = 0xD4
pub const NV_ESC_QUERY_DEVICE_INTR: u32 = NV_IOCTL_BASE + 13; // 213 = 0xD5
pub const NV_ESC_SYS_PARAMS: u32 = NV_IOCTL_BASE + 14; // 214 = 0xD6
pub const NV_ESC_EXPORT_TO_DMABUF_FD: u32 = NV_IOCTL_BASE + 17; // 217 = 0xD9
pub const NV_ESC_WAIT_OPEN_COMPLETE: u32 = NV_IOCTL_BASE + 18; // 218 = 0xDA

// ---------------------------------------------------------------------------
// RM escapes — from nv_escape.h
// ---------------------------------------------------------------------------

pub const NV_ESC_RM_ALLOC_MEMORY: u32 = 0x27;
pub const NV_ESC_RM_ALLOC_OBJECT: u32 = 0x28;
pub const NV_ESC_RM_FREE: u32 = 0x29;
pub const NV_ESC_RM_CONTROL: u32 = 0x2A;
pub const NV_ESC_RM_ALLOC: u32 = 0x2B;
pub const NV_ESC_RM_DUP_OBJECT: u32 = 0x34;
pub const NV_ESC_RM_SHARE: u32 = 0x35;
pub const NV_ESC_RM_I2C_ACCESS: u32 = 0x39;
pub const NV_ESC_RM_IDLE_CHANNELS: u32 = 0x41;
pub const NV_ESC_RM_VID_HEAP_CONTROL: u32 = 0x4A;
pub const NV_ESC_RM_ACCESS_REGISTRY: u32 = 0x4D;
pub const NV_ESC_RM_MAP_MEMORY: u32 = 0x4E;
pub const NV_ESC_RM_UNMAP_MEMORY: u32 = 0x4F;
pub const NV_ESC_RM_GET_EVENT_DATA: u32 = 0x52;
pub const NV_ESC_RM_ALLOC_CONTEXT_DMA2: u32 = 0x54;
pub const NV_ESC_RM_ADD_VBLANK_CALLBACK: u32 = 0x56;
pub const NV_ESC_RM_MAP_MEMORY_DMA: u32 = 0x57;
pub const NV_ESC_RM_UNMAP_MEMORY_DMA: u32 = 0x58;
pub const NV_ESC_RM_BIND_CONTEXT_DMA: u32 = 0x59;
pub const NV_ESC_RM_EXPORT_OBJECT_TO_FD: u32 = 0x5C;
pub const NV_ESC_RM_IMPORT_OBJECT_FROM_FD: u32 = 0x5D;
pub const NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO: u32 = 0x5E;
pub const NV_ESC_RM_LOCKLESS_DIAGNOSTIC: u32 = 0x5F;

// ---------------------------------------------------------------------------
// /dev/nvidia-uvm ioctl numbers  (Phase 4)
// ---------------------------------------------------------------------------

pub const UVM_INITIALIZE: u32 = 0x01;
pub const UVM_DEINITIALIZE: u32 = 0x02;

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
