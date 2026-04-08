// src/devices/src/virtio/gpu_nv/virtio.rs
//
// VirtioDevice trait implementation for GpuNv, plus the virtqueue event loop.

use std::sync::atomic::Ordering;
use vm_memory::{Bytes, GuestMemoryMmap};

use crate::virtio::gpu_nv::{GpuNv, VIRTIO_ID_GPU_NV};
use crate::virtio::{
    ActivateError, DeviceQueue, InterruptTransport, QueueConfig, VirtioDevice,
    VIRTIO_MMIO_INT_VRING,
};

// ─────────────────────────────────────────────────────────────────────────────
// VirtioDevice trait
// ─────────────────────────────────────────────────────────────────────────────

impl VirtioDevice for GpuNv {
    fn avail_features(&self) -> u64 {
        self.avail_features
    }

    fn acked_features(&self) -> u64 {
        self.acked_features
    }

    fn set_acked_features(&mut self, acked_features: u64) {
        self.acked_features = acked_features & self.avail_features;
    }
    fn device_type(&self) -> u32 {
        VIRTIO_ID_GPU_NV
    }
    fn device_name(&self) -> &str {
        "virtio-nvgpu"
    }

    fn queue_config(&self) -> &[QueueConfig] {
        // Two queues: controlq (index 0) and eventq (index 1), both max size 256
        const QUEUE_CFGS: [QueueConfig; 2] = [QueueConfig::new(256), QueueConfig::new(256)];
        &QUEUE_CFGS
    }

    fn avail_features_by_page(&self, _page: u32) -> u32 {
        // Feature bits 0-2 in page 0
        use crate::virtio::gpu_nv::{NVGPU_CAP_COMPUTE, NVGPU_CAP_GRAPHICS, NVGPU_CAP_VIDEO};
        let caps = self.config.caps;
        let mut bits: u32 = 0;
        if caps & NVGPU_CAP_COMPUTE != 0 {
            bits |= 1 << 0;
        } // F_UVM
        if caps & NVGPU_CAP_VIDEO != 0 {
            bits |= 1 << 1;
        } // F_ENCODE
        if caps & NVGPU_CAP_GRAPHICS != 0 {
            bits |= 1 << 2;
        } // F_GRAPHICS
        bits
    }

    fn ack_features_by_page(&mut self, _page: u32, _value: u32) {
        // Accept whatever the guest negotiates; we expose all features.
    }

    fn read_config(&self, offset: u64, data: &mut [u8]) {
        let cfg = self.config_bytes();
        let start = offset as usize;
        let end = std::cmp::min(start + data.len(), cfg.len());
        if start < end {
            data[..end - start].copy_from_slice(&cfg[start..end]);
        }
    }

    fn write_config(&mut self, _offset: u64, _data: &[u8]) {
        // Config space is read-only from the guest's perspective.
    }

    fn activate(
        &mut self,
        mem: GuestMemoryMmap,
        interrupt: InterruptTransport, // Update this type
        queues: Vec<DeviceQueue>,
    ) -> Result<(), ActivateError> {
        self.guest_memory = Some(mem);
        self.interrupt_transport = Some(interrupt);
        self.queues = queues;

        // Ensure we have the expected number of queues
        if self.queues.len() != 2 {
            return Err(ActivateError::BadActivate);
        }

        Ok(())
    }

    fn is_activated(&self) -> bool {
        self.guest_memory.is_some()
    }

    fn reset(&mut self) -> bool {
        // Deactivate: drop queues and memory reference
        self.queues.clear();
        self.interrupt_transport = None;
        self.guest_memory = None;
        self.acked_features = 0;
        true
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Virtqueue polling — called by the VMM event loop when controlq is kicked
// ─────────────────────────────────────────────────────────────────────────────

impl GpuNv {
    /// Process all pending descriptors on the control virtqueue.
    ///
    /// This should be called every time the VMM receives a kick event on the
    /// controlq eventfd.  It is synchronous: each request is fully processed
    /// before moving to the next.
    pub fn process_controlq(&mut self) {
        let mem = match self.guest_memory.clone() {
            Some(m) => m,
            None => return,
        };

        loop {
            let (desc_chain, head_index) = {
                let queue = &mut self.queues[0].queue;
                match queue.pop(&mem) {
                    Some(dc) => {
                        let idx = dc.index;
                        (dc, idx)
                    }
                    None => break,
                }
            };

            // ── Collect all readable descriptor data (the request) ───────────
            let mut req_buf: Vec<u8> = Vec::new();
            // --- Collect writable descriptors for the response ---
            let mut write_descs = Vec::new();
            for desc in desc_chain.into_iter() {
                if desc.is_read_only() {
                    let len = desc.len as usize;
                    let mut chunk = vec![0u8; len];
                    if mem.read_slice(&mut chunk, desc.addr).is_ok() {
                        req_buf.extend_from_slice(&chunk);
                    }
                } else if desc.is_write_only() {
                    write_descs.push((desc.addr, desc.len));
                }
            }

            // ── Dispatch ─────────────────────────────────────────────────────
            let resp_buf = self.process_request(&req_buf);

            // ── Write response into writable descriptors ─────────────────────
            let mut written = 0usize;
            for (addr, len) in write_descs {
                if written >= resp_buf.len() {
                    break;
                }
                let avail = len as usize;
                let to_write = std::cmp::min(avail, resp_buf.len() - written);
                let _ = mem.write_slice(&resp_buf[written..written + to_write], addr);
                written += to_write;
            }

            {
                let queue = &mut self.queues[0].queue;
                let _ = queue.add_used(&mem, head_index, written as u32);
            }
        }

        // Signal the guest that the used ring has been updated.
        if let Some(transport) = self.interrupt_transport.as_ref() {
            transport
                .status()
                .fetch_or(VIRTIO_MMIO_INT_VRING as usize, Ordering::SeqCst);
            let _ = transport.event().write(1);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Config space serialisation
    // ─────────────────────────────────────────────────────────────────────────

    fn config_bytes(&self) -> Vec<u8> {
        // Layout matches struct virtio_gpu_nv_config in the C driver:
        //   char     driver_version[32];   // 0..32
        //   uint32_t num_gpus;             // 32..36
        //   uint32_t caps;                 // 36..40
        //   uint32_t gpu_device_ids[8];    // 40..72
        let mut buf = vec![0u8; 72];

        let ver = self.config.driver_version.as_bytes();
        let copy_len = std::cmp::min(ver.len(), 31); // leave NUL terminator
        buf[..copy_len].copy_from_slice(&ver[..copy_len]);

        buf[32..36].copy_from_slice(&self.config.num_gpus.to_le_bytes());
        buf[36..40].copy_from_slice(&self.config.caps.to_le_bytes());
        // gpu_device_ids: fill with sequential IDs 0..num_gpus
        for i in 0..std::cmp::min(self.config.num_gpus as usize, 8) {
            let off = 40 + i * 4;
            buf[off..off + 4].copy_from_slice(&(i as u32).to_le_bytes());
        }

        buf
    }
}
