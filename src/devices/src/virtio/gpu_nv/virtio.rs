// src/devices/src/virtio/gpu_nv/virtio.rs
//
// VirtioDevice trait implementation for GpuNv, plus the virtqueue event loop.

use crate::virtio::gpu_nv::device::MmioAllocator;
use crate::virtio::gpu_nv::worker::Worker;
use crate::virtio::gpu_nv::{GpuNv, VIRTIO_ID_GPU_NV};
use crate::virtio::{ActivateError, DeviceQueue, InterruptTransport, QueueConfig, VirtioDevice};
use vm_memory::GuestMemoryMmap;

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

    fn avail_features_by_page(&self, page: u32) -> u32 {
        // Feature bits 0-2 in page 0
        use crate::virtio::gpu_nv::{NVGPU_CAP_COMPUTE, NVGPU_CAP_GRAPHICS, NVGPU_CAP_VIDEO};
        match page {
            0 => {
                let caps = self.config.caps;
                let mut bits: u32 = 0;
                if caps & NVGPU_CAP_COMPUTE != 0 {
                    bits |= 1 << 0; // F_UVM
                }
                if caps & NVGPU_CAP_VIDEO != 0 {
                    bits |= 1 << 1; // F_ENCODE
                }
                if caps & NVGPU_CAP_GRAPHICS != 0 {
                    bits |= 1 << 2; // F_GRAPHICS
                }
                bits
            }
            1 => 1 << 0, // VIRTIO_F_VERSION_1 = bit 32, so bit 0 of page 1
            _ => 0,
        }
    }

    fn ack_features_by_page(&mut self, _page: u32, _value: u32) {
        // Accept whatever the guest negotiates; we expose all features.
    }

    fn read_config(&self, offset: u64, data: &mut [u8]) {
        let cfg = self.config_bytes();
        let start = offset as usize;
        let end = (start + data.len()).min(cfg.len());
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
        interrupt: InterruptTransport,
        queues: Vec<DeviceQueue>,
    ) -> Result<(), ActivateError> {
        let [control_q, _event_q]: [_; 2] = queues.try_into().map_err(|_| {
            log::error!("virtio-gpu-nv: expected 2 queues");
            ActivateError::BadActivate
        })?;

        // Move all runtime state into the worker — it owns everything now.
        let worker = Worker::new(
            control_q,
            mem.clone(),
            interrupt.clone(),
            self.config.clone(),
            std::mem::take(&mut self.fd_table),
            self.next_handle,
            std::mem::take(&mut self.mappings),
            self.next_mapping_id,
            std::mem::replace(&mut self.mmio_alloc, MmioAllocator::new(0, 0)),
            self.vm_fd.clone(),
            self.next_kvm_slot,
            self.allowed_ioctls.clone(),
        );

        // Spawn the worker thread — same pattern as the existing GPU device.
        worker.run();

        // Keep references for is_activated() / reset()
        self.guest_memory = Some(mem);
        self.interrupt_transport = Some(interrupt);

        Ok(())
    }

    fn is_activated(&self) -> bool {
        self.guest_memory.is_some()
    }

    fn reset(&mut self) -> bool {
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
    // ─────────────────────────────────────────────────────────────────────────
    // Config space serialisation
    // ─────────────────────────────────────────────────────────────────────────

    fn config_bytes(&self) -> Vec<u8> {
        // Fixed layout — always exactly this size regardless of num_gpus.
        // Guest struct offsets are therefore compile-time constants.
        //
        //   [    0..   32] driver_version[32]
        //   [   32..   36] num_gpus
        //   [   36..   40] caps
        //   [   40..   72] gpu_device_ids[8]
        //   [   72.. 8776] gpu_slots[8]        — 8 * 1088, unused slots zeroed
        //   [ 8776.. 8780] num_fd_translations
        //   [ 8780.. 8784] _pad
        //   [ 8784.. 8912] fd_translations[16] — 16 * 8, unused entries zeroed

        const GPU_SLOT: usize = 1088;
        const N_GPU_SLOTS: usize = 8;
        const N_FD_SLOTS: usize = 16;
        const TOTAL: usize = 72
            + N_GPU_SLOTS * GPU_SLOT   // 8704
            + 4                        // num_fd_translations
            + 4                        // _pad
            + N_FD_SLOTS * 8; // 128
                              // TOTAL = 8912

        let mut buf = vec![0u8; TOTAL];

        // driver_version
        let ver = self.config.driver_version.as_bytes();
        let vlen = ver.len().min(31);
        buf[..vlen].copy_from_slice(&ver[..vlen]);

        // num_gpus — actual count, not 8
        let n = self.config.gpus.len().min(N_GPU_SLOTS);
        buf[32..36].copy_from_slice(&(n as u32).to_le_bytes());

        // caps
        buf[36..40].copy_from_slice(&self.config.caps.to_le_bytes());

        // gpu_device_ids
        for i in 0..n {
            let off = 40 + i * 4;
            buf[off..off + 4].copy_from_slice(&(i as u32).to_le_bytes());
        }

        // GPU slots — only populate actual GPUs, rest stay zero
        for (i, gpu) in self.config.gpus.iter().take(N_GPU_SLOTS).enumerate() {
            let base = 72 + i * GPU_SLOT;

            let s = gpu.pci_addr.as_bytes();
            let l = s.len().min(15);
            buf[base..base + l].copy_from_slice(&s[..l]);

            buf[base + 16..base + 20].copy_from_slice(&gpu.minor.to_le_bytes());

            let text = gpu.information.content.as_bytes();
            let tlen = text.len().min(1059);
            buf[base + 20..base + 24].copy_from_slice(&(tlen as u32).to_le_bytes());
            buf[base + 28..base + 28 + tlen].copy_from_slice(&text[..tlen]);
        }

        // fd-translation table — at fixed offset 8776
        const FD_TABLE_BASE: usize = 72 + N_GPU_SLOTS * GPU_SLOT;
        let fd_entries = &crate::virtio::gpu_nv::allowlist::FD_TRANSLATION_IOCTLS;
        let nf = fd_entries.len().min(N_FD_SLOTS);
        buf[FD_TABLE_BASE..FD_TABLE_BASE + 4].copy_from_slice(&(nf as u32).to_le_bytes());
        // [FD_TABLE_BASE+4..+8] stays zero (_pad)
        for (i, entry) in fd_entries.iter().take(N_FD_SLOTS).enumerate() {
            let base = FD_TABLE_BASE + 8 + i * 8;
            buf[base..base + 4].copy_from_slice(&entry.nr.to_le_bytes());
            buf[base + 4..base + 8].copy_from_slice(&(entry.payload_offset as u32).to_le_bytes());
        }

        buf
    }
}
