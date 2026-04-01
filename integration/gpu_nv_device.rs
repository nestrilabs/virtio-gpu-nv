// crates/device/src/virtio.rs
//
// Virtio device implementation for libkrun integration.
//
// Implements the `VirtioDevice` trait from libkrun's device infrastructure.
// The device presents as VIRTIO_ID_GPU_NV (0x8042) with a single request
// virtqueue.  On activation, a worker thread is spawned that polls the
// queue and dispatches requests through NvidiaBackend.
//
// The SHM BAR (backed by memfd) is exposed to the guest via the
// VirtioShmRegion mechanism, which libkrun maps into the guest's
// physical address space as a KVM memslot.

use std::io::{Read, Write};
use std::os::fd::{AsRawFd, BorrowedFd};
use std::sync::{Arc, Mutex};
use std::thread;

use tracing::{error, warn};

use nix::fcntl::{fcntl, FcntlArg, OFlag};
use vm_memory::GuestMemoryMmap;

use crate::nvidia::NvidiaBackend;
use crate::shm::ZoneConfig;

// ---------------------------------------------------------------------------
// Re-export libkrun virtio types used by our public API
// ---------------------------------------------------------------------------

use devices::virtio::{
    ActivateError, ActivateResult, DeviceQueue, DeviceState, InterruptTransport,
    Queue as VirtQueue, QueueConfig, VirtioDevice, VirtioShmRegion,
};
use devices::virtio::descriptor_utils::{Reader, Writer};
use devices::virtio::AsAny;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Vendor-specific virtio device ID for virtio-gpu-nv.
pub const VIRTIO_ID_GPU_NV: u32 = 0x8042;

/// We use a single request/response virtqueue.
const NUM_QUEUES: usize = 1;

/// virtio feature: version 1
const VIRTIO_F_VERSION_1: u64 = 1 << 32;

/// Queue size — 256 entries is standard for most virtio devices.
const QUEUE_SIZE: u16 = 256;

static QUEUE_CONFIG: [QueueConfig; NUM_QUEUES] = [QueueConfig::new(QUEUE_SIZE)];

/// Device configuration space.
///
/// Exposed to the guest via read_config(). The guest driver can read the
/// SHM BAR GPA from here to set up nv_mmap().
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
struct NvGpuConfig {
    /// Number of GPU devices available (0..MAX_GPU).
    num_gpus: u32,
    /// Guest-physical address of the SHM BAR (set by VMM after memslot creation).
    shm_bar_gpa: u64,
    /// Total size of the SHM BAR in bytes.
    shm_bar_size: u64,
}

// ---------------------------------------------------------------------------
// NvGpuDevice — the VirtioDevice implementation
// ---------------------------------------------------------------------------

pub struct NvGpuDevice {
    avail_features: u64,
    acked_features: u64,
    device_state: DeviceState,
    shm_region: Option<VirtioShmRegion>,
    backend: Arc<Mutex<NvidiaBackend>>,
    config: NvGpuConfig,
}

impl NvGpuDevice {
    pub fn new(cfg: ZoneConfig) -> Self {
        let backend = NvidiaBackend::new(cfg);
        let shm_size = backend.shm_total_size();

        Self {
            avail_features: VIRTIO_F_VERSION_1,
            acked_features: 0,
            device_state: DeviceState::Inactive,
            shm_region: None,
            config: NvGpuConfig {
                num_gpus: 1,
                shm_bar_gpa: 0,
                shm_bar_size: shm_size,
            },
            backend: Arc::new(Mutex::new(backend)),
        }
    }

    pub fn with_default_zones() -> Self {
        Self::new(ZoneConfig::default_256mib())
    }

    /// Set the SHM region after the VMM has created the KVM memslot.
    /// Must be called before the guest boots.
    pub fn set_shm_region(&mut self, region: VirtioShmRegion) {
        self.config.shm_bar_gpa = region.guest_addr;
        self.config.shm_bar_size = region.size as u64;
        self.shm_region = Some(region);
    }

    /// Get the memfd raw fd for KVM memslot creation.
    pub fn shm_memfd_raw(&self) -> i32 {
        self.backend.lock().unwrap().shm_memfd_raw()
    }

    /// Teardown: close all host fds.
    pub fn teardown(&self) {
        self.backend.lock().unwrap().teardown();
    }
}

// AsAny is required by libkrun's VirtioDevice trait
impl AsAny for NvGpuDevice {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl VirtioDevice for NvGpuDevice {
    fn avail_features(&self) -> u64 {
        self.avail_features
    }

    fn acked_features(&self) -> u64 {
        self.acked_features
    }

    fn set_acked_features(&mut self, acked_features: u64) {
        self.acked_features = acked_features;
    }

    fn device_type(&self) -> u32 {
        VIRTIO_ID_GPU_NV
    }

    fn device_name(&self) -> &str {
        "gpu-nv"
    }

    fn queue_config(&self) -> &[QueueConfig] {
        &QUEUE_CONFIG
    }

    fn read_config(&self, offset: u64, mut data: &mut [u8]) {
        let config_bytes = unsafe {
            std::slice::from_raw_parts(
                &self.config as *const NvGpuConfig as *const u8,
                std::mem::size_of::<NvGpuConfig>(),
            )
        };
        let config_len = config_bytes.len() as u64;
        if offset >= config_len {
            error!("gpu-nv: read_config: offset {} out of bounds", offset);
            return;
        }
        let end = std::cmp::min(offset + data.len() as u64, config_len) as usize;
        let _ = data.write_all(&config_bytes[offset as usize..end]);
    }

    fn write_config(&mut self, offset: u64, data: &[u8]) {
        warn!(
            "gpu-nv: guest attempted to write config (offset={:#x}, len={})",
            offset,
            data.len()
        );
    }

    fn activate(
        &mut self,
        mem: GuestMemoryMmap,
        interrupt: InterruptTransport,
        queues: Vec<DeviceQueue>,
    ) -> ActivateResult {
        let [request_q]: [_; NUM_QUEUES] = queues.try_into().map_err(|_| {
            error!("gpu-nv: expected {} queue(s)", NUM_QUEUES);
            ActivateError::BadActivate
        })?;

        let backend = self.backend.clone();
        let worker = NvWorker::new(request_q, mem.clone(), interrupt.clone(), backend);
        worker.run();

        self.device_state = DeviceState::Activated(mem, interrupt);
        Ok(())
    }

    fn is_activated(&self) -> bool {
        self.device_state.is_activated()
    }

    fn reset(&mut self) -> bool {
        self.teardown();
        self.device_state = DeviceState::Inactive;
        true
    }

    fn shm_region(&self) -> Option<&VirtioShmRegion> {
        self.shm_region.as_ref()
    }
}

// ---------------------------------------------------------------------------
// NvWorker — processes virtqueue entries on a dedicated thread
// ---------------------------------------------------------------------------

struct NvWorker {
    queue_evt: utils::eventfd::EventFd,
    queue: Arc<Mutex<VirtQueue>>,
    mem: GuestMemoryMmap,
    interrupt: InterruptTransport,
    backend: Arc<Mutex<NvidiaBackend>>,
}

impl NvWorker {
    fn new(
        request_q: DeviceQueue,
        mem: GuestMemoryMmap,
        interrupt: InterruptTransport,
        backend: Arc<Mutex<NvidiaBackend>>,
    ) -> Self {
        // Clone the eventfd and set to blocking mode so read() blocks
        // until the guest kicks the queue.
        let queue_evt = request_q.event.try_clone().unwrap();
        let fd = unsafe { BorrowedFd::borrow_raw(queue_evt.as_raw_fd()) };
        let flags =
            OFlag::from_bits_retain(fcntl(fd, FcntlArg::F_GETFL).unwrap()) & !OFlag::O_NONBLOCK;
        fcntl(fd, FcntlArg::F_SETFL(flags)).unwrap();

        Self {
            queue_evt,
            queue: Arc::new(Mutex::new(request_q.queue)),
            mem,
            interrupt,
            backend,
        }
    }

    fn run(self) {
        thread::Builder::new()
            .name("gpu-nv worker".into())
            .spawn(|| self.work())
            .unwrap();
    }

    fn work(self) {
        loop {
            // Block until the guest kicks the queue.
            if let Err(e) = self.queue_evt.read() {
                error!("gpu-nv: failed to read queue eventfd: {:?}", e);
                continue;
            }

            if self.process_queue() {
                if let Err(e) = self.interrupt.try_signal_used_queue() {
                    error!("gpu-nv: failed to signal used queue: {:?}", e);
                }
            }
        }
    }

    fn process_queue(&self) -> bool {
        let mut used_any = false;

        loop {
            let head = self.queue.lock().unwrap().pop(&self.mem);

            let Some(head) = head else {
                break;
            };

            let desc_index = head.index;

            // Read the entire request from the readable descriptors.
            let mut reader = match Reader::new(&self.mem, head.clone()) {
                Ok(r) => r,
                Err(e) => {
                    error!("gpu-nv: failed to create Reader: {:?}", e);
                    continue;
                }
            };

            let mut writer = match Writer::new(&self.mem, head) {
                Ok(w) => w,
                Err(e) => {
                    error!("gpu-nv: failed to create Writer: {:?}", e);
                    continue;
                }
            };

            // Read all readable bytes into a contiguous buffer.
            let req_len = reader.available_bytes();
            let mut req_buf = vec![0u8; req_len];
            if let Err(e) = reader.read_exact(&mut req_buf) {
                error!("gpu-nv: failed to read request: {:?}", e);
                continue;
            }

            // Dispatch through the backend.
            let resp_capacity = writer.available_bytes();
            let mut resp_buf = vec![0u8; resp_capacity];
            let resp_len = {
                let mut backend = self.backend.lock().unwrap();
                backend.dispatch(&req_buf, &mut resp_buf)
            };

            // Write the response into the writable descriptors.
            if resp_len > 0 {
                if let Err(e) = writer.write_all(&resp_buf[..resp_len]) {
                    error!("gpu-nv: failed to write response: {:?}", e);
                }
            }

            let written = writer.bytes_written() as u32;
            if let Err(e) = self.queue.lock().unwrap().add_used(
                &self.mem,
                desc_index,
                written,
            ) {
                error!("gpu-nv: failed to add used: {:?}", e);
            }

            used_any = true;
        }

        used_any
    }
}
