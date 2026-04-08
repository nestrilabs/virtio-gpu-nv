// src/devices/src/virtio/gpu_nv/allowlist.rs
//
// Security boundary: only ioctls on this list are forwarded to the host
// NVIDIA driver.  Unknown commands are rejected with ENOTTY.
//
// The frontend list is ported from gVisor's nvproxy frontendIoctl map.
// UVM ioctls are always forwarded for the full UVM command set.

use std::collections::HashSet;

use crate::virtio::gpu_nv::{NVGPU_CAP_GRAPHICS, NVGPU_CAP_VIDEO};

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

        // ── Always-allowed frontend ioctls (compute + utility) ───────────────
        let base: &[u32] = &[
            0x01, // NV_ESC_CARD_INFO (alternate nr seen on some versions)
            0x27, // NV_ESC_RM_ALLOC_CONTEXT_DMA2
            0x29, // NV_ESC_RM_FREE
            0x2a, // NV_ESC_RM_CONTROL
            0x2b, // NV_ESC_RM_ALLOC
            0x2e, // NV_ESC_RM_MAP_MEMORY
            0x2f, // NV_ESC_RM_UNMAP_MEMORY
            0x34, // NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO
            0x37, // NV_ESC_REGISTER_FD
            0x4d, // NV_ESC_CARD_INFO
            0x57, // NV_ESC_SYS_PARAMS
            0x58, // NV_ESC_NUMA_INFO
        ];
        for &nr in base {
            a.frontend.insert(nr);
        }

        // ── Graphics capability ──────────────────────────────────────────────
        if caps & NVGPU_CAP_GRAPHICS != 0 {
            a.frontend.insert(0x27); // NV_ESC_RM_ALLOC_CONTEXT_DMA2
            a.frontend.insert(0x30); // NV_ESC_RM_IDLE_CHANNELS
            a.frontend.insert(0x31); // NV_ESC_RM_MAP_MEMORY_DMA
            a.frontend.insert(0x32); // NV_ESC_RM_UNMAP_MEMORY_DMA
            a.frontend.insert(0x35); // NV_ESC_RM_BIND_CONTEXT_DMA
        }

        // ── Video capability (NVENC / NVDEC) ─────────────────────────────────
        if caps & NVGPU_CAP_VIDEO != 0 {
            a.frontend.insert(0x31); // NV_ESC_RM_MAP_MEMORY_DMA
            a.frontend.insert(0x32); // NV_ESC_RM_UNMAP_MEMORY_DMA
        }

        // ── UVM ioctls (all forwarded; UVM is an isolated device) ────────────
        //
        // UVM numbers 0x00-0x3f; list the most common ones explicitly.
        // In practice we allow all UVM ioctls since UVM does its own
        // per-client isolation and has a different security model.
        for nr in 0x00u32..=0x3fu32 {
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
}
