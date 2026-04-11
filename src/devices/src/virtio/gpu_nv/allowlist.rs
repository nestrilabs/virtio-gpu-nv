// src/devices/src/virtio/gpu_nv/allowlist.rs
//
// Security boundary: only ioctls on this list are forwarded to the host
// NVIDIA driver.  Unknown commands are rejected with ENOTTY.
//
// The frontend list is ported from gVisor's nvproxy frontendIoctl map.
// UVM ioctls are always forwarded for the full UVM command set.

use std::collections::HashSet;

use crate::virtio::gpu_nv::{NVGPU_CAP_GRAPHICS, NVGPU_CAP_VIDEO};

/// Ioctls whose payload contains a guest fd that must be translated to a
/// host fd number before forwarding.  Add new entries here as they are
/// discovered — no other code needs to change.
///
/// `payload_offset` is the byte offset within the ioctl data buffer where
/// the 4-byte fd value lives.
#[derive(Clone, Copy)]
pub struct FdTranslationEntry {
    pub nr: u32,
    pub payload_offset: usize,
}

/// All frontend ioctls that carry an embedded fd operand.
pub const FD_TRANSLATION_IOCTLS: &[FdTranslationEntry] = &[
    FdTranslationEntry {
        nr: 0xc9,
        payload_offset: 0,
    }, // NV_ESC_REGISTER_FD
];

#[derive(Clone)]
pub struct AllowedIoctls {
    /// Allowed IOC_NR values for /dev/nvidia* and /dev/nvidiactl.
    frontend: HashSet<u32>,
    /// Allowed IOC_NR values for /dev/nvidia-uvm.
    uvm: HashSet<u32>,
}
impl AllowedIoctls {
    pub fn new(caps: u32) -> Self {
        let mut a = AllowedIoctls {
            frontend: HashSet::new(),
            uvm: HashSet::new(),
        };

        const NV_IOCTL_BASE: u32 = 200;

        // ── Always-allowed frontend ioctls (compute + utility) ───────────────
        let base: &[u32] = &[
            0x01,               // NV_ESC_CARD_INFO (alternate nr seen on some versions)
            0x02,               // unsure.. but nvidia-smi calls it
            0x54,               // NV_ESC_RM_ALLOC_CONTEXT_DMA2
            0x29,               // NV_ESC_RM_FREE
            0x2A,               // NV_ESC_RM_CONTROL
            0x2B,               // NV_ESC_RM_ALLOC
            0x4E,               // NV_ESC_RM_MAP_MEMORY
            0x4F,               // NV_ESC_RM_UNMAP_MEMORY
            0x5E,               // NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO
            0x58,               // NV_ESC_NUMA_INFO
            0xC8,               // NV_ESC_QUERY_DEVICE_INTR
            0xD2,               // NV_ESC_CHECK_VERSION_STR
            0xD6,               // NV_ESC_WAIT_OPEN_COMPLETE
            0xDa,               // NV_ESC_ATTACH_GPUS_TO_FD
            NV_IOCTL_BASE + 0,  // NV_ESC_CARD_INFO           0xC8
            NV_IOCTL_BASE + 1,  // NV_ESC_REGISTER_FD          0xC9
            NV_IOCTL_BASE + 6,  // NV_ESC_ALLOC_OS_EVENT       0xCE
            NV_IOCTL_BASE + 7,  // NV_ESC_FREE_OS_EVENT        0xCF
            NV_IOCTL_BASE + 9,  // NV_ESC_STATUS_CODE          0xD1
            NV_IOCTL_BASE + 10, // NV_ESC_CHECK_VERSION_STR    0xD2
            NV_IOCTL_BASE + 11, // NV_ESC_IOCTL_XFER_CMD       0xD3
            NV_IOCTL_BASE + 12, // NV_ESC_ATTACH_GPUS_TO_FD    0xD4
            NV_IOCTL_BASE + 13, // NV_ESC_QUERY_DEVICE_INTR    0xD5
            NV_IOCTL_BASE + 14, // NV_ESC_SYS_PARAMS           0xD6
            NV_IOCTL_BASE + 15, // undocumented in 595.58.03   0xD7
            NV_IOCTL_BASE + 16, // undocumented                0xD8
            NV_IOCTL_BASE + 17, // NV_ESC_EXPORT_TO_DMABUF_FD  0xD9
            NV_IOCTL_BASE + 18, // NV_ESC_WAIT_OPEN_COMPLETE   0xDA
        ];
        for &nr in base {
            a.frontend.insert(nr);
        }

        // ── Graphics capability ──────────────────────────────────────────────
        if caps & NVGPU_CAP_GRAPHICS != 0 {
            a.frontend.insert(0x27); // NV_ESC_RM_ALLOC_CONTEXT_DMA2
            a.frontend.insert(0x41); // NV_ESC_RM_IDLE_CHANNELS
            a.frontend.insert(0x34); // TODO: Write which one is this
            a.frontend.insert(0x4a); // TODO: Write which one is this
            a.frontend.insert(0x57); // NV_ESC_RM_MAP_MEMORY_DMA
            a.frontend.insert(0x58); // NV_ESC_RM_UNMAP_MEMORY_DMA
            a.frontend.insert(0x59); // NV_ESC_RM_BIND_CONTEXT_DMA
        }

        // ── Video capability (NVENC / NVDEC) ─────────────────────────────────
        if caps & NVGPU_CAP_VIDEO != 0 {
            a.frontend.insert(0x57); // NV_ESC_RM_MAP_MEMORY_DMA
            a.frontend.insert(0x58); // NV_ESC_RM_UNMAP_MEMORY_DMA
        }

        // ── UVM ioctls (all forwarded; UVM is an isolated device) ────────────
        //
        // UVM numbers 0x00-0x4f; list the most common ones explicitly.
        // In practice we allow all UVM ioctls since UVM does its own
        // per-client isolation and has a different security model.
        for nr in 0x00u32..=0x4fu32 {
            a.uvm.insert(nr);
        }

        a
    }

    /// Returns true if the ioctl IOC_NR is allowed on frontend devices.
    pub fn is_allowed(&self, nr: u32) -> bool {
        self.frontend.contains(&nr)
    }

    /// Returns true if the ioctl command is allowed on /dev/nvidia-uvm.
    pub fn is_uvm_allowed(&self, cmd: u32) -> bool {
        let nr = cmd & 0xFF;
        self.uvm.contains(&nr)
    }

    /// If `nr` is an fd-carrying ioctl, return its translation entry.
    pub fn fd_translation(nr: u32) -> Option<&'static FdTranslationEntry> {
        FD_TRANSLATION_IOCTLS.iter().find(|e| e.nr == nr)
    }
}
