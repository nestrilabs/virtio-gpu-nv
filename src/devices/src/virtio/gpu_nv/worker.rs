// src/devices/src/virtio/gpu_nv/worker.rs

use std::collections::HashMap;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, BorrowedFd};
use std::sync::{Arc, Mutex};
use std::thread;

use nix::fcntl::{fcntl, FcntlArg, OFlag};
use utils::eventfd::EventFd;
use vm_memory::GuestMemoryMmap;

use crate::virtio::descriptor_utils::{Reader, Writer};
use crate::virtio::gpu_nv::allowlist::AllowedIoctls;
use crate::virtio::gpu_nv::device::{GpuMapping, GpuNvConfig, HostFd, MmioAllocator};
use crate::virtio::{DeviceQueue, InterruptTransport, Queue as VirtQueue};

pub(crate) struct Worker {
    control_evt: EventFd,
    control_queue: Arc<Mutex<VirtQueue>>,
    mem: GuestMemoryMmap,
    interrupt: InterruptTransport,

    // Runtime state — moved from GpuNv during activate()
    pub(crate) config: GpuNvConfig,
    pub(crate) fd_table: HashMap<u32, HostFd>,
    pub(crate) next_handle: u32,
    pub(crate) mappings: HashMap<u32, GpuMapping>,
    pub(crate) next_mapping_id: u32,
    pub(crate) mmio_alloc: MmioAllocator,
    pub(crate) vm_fd: Arc<kvm_ioctls::VmFd>,
    pub(crate) next_kvm_slot: u32,
    pub(crate) allowed_ioctls: AllowedIoctls,
}

impl Worker {
    pub fn new(
        control_q: DeviceQueue,
        mem: GuestMemoryMmap,
        interrupt: InterruptTransport,
        config: GpuNvConfig,
        fd_table: HashMap<u32, HostFd>,
        next_handle: u32,
        mappings: HashMap<u32, GpuMapping>,
        next_mapping_id: u32,
        mmio_alloc: MmioAllocator,
        vm_fd: Arc<kvm_ioctls::VmFd>,
        next_kvm_slot: u32,
        allowed_ioctls: AllowedIoctls,
    ) -> Self {
        // Clone the eventfd and set it to blocking mode — exactly like the
        // existing GPU worker does.
        let control_evt = control_q.event.try_clone().unwrap();
        let fd = unsafe { BorrowedFd::borrow_raw(control_evt.as_raw_fd()) };
        let flags =
            OFlag::from_bits_retain(fcntl(fd, FcntlArg::F_GETFL).unwrap()) & !OFlag::O_NONBLOCK;
        fcntl(fd, FcntlArg::F_SETFL(flags)).unwrap();

        Self {
            control_evt,
            control_queue: Arc::new(Mutex::new(control_q.queue)),
            mem,
            interrupt,
            config,
            fd_table,
            next_handle,
            mappings,
            next_mapping_id,
            mmio_alloc,
            vm_fd,
            next_kvm_slot,
            allowed_ioctls,
        }
    }

    pub fn run(self) {
        thread::Builder::new()
            .name("gpu_nv worker".into())
            .spawn(|| self.work())
            .unwrap();
    }

    fn work(mut self) {
        loop {
            if let Err(e) = self.control_evt.read() {
                error!("gpu_nv: control_evt read error: {:?}", e);
                continue;
            }
            if self.process_controlq() {
                if let Err(e) = self.interrupt.try_signal_used_queue() {
                    error!("gpu_nv: error signaling used queue: {:?}", e);
                }
            }
        }
    }

    fn process_controlq(&mut self) -> bool {
        let mem = self.mem.clone();
        let mut used_any = false;

        loop {
            let head = match self.control_queue.lock().unwrap().pop(&mem) {
                Some(h) => h,
                None => break,
            };

            let head_index = head.index;

            // Use Reader/Writer like the existing GPU worker — don't manually
            // iterate descriptors
            let mut reader = match Reader::new(&mem, head.clone()) {
                Ok(r) => r,
                Err(e) => {
                    error!("gpu_nv: reader error: {:?}", e);
                    continue;
                }
            };
            let mut writer = match Writer::new(&mem, head.clone()) {
                Ok(w) => w,
                Err(e) => {
                    error!("gpu_nv: writer error: {:?}", e);
                    continue;
                }
            };

            // Read entire request into a Vec
            let req_len = reader.available_bytes();
            let mut req_buf = vec![0u8; req_len];
            if let Err(e) = reader.read_exact(&mut req_buf) {
                error!("gpu_nv: failed to read request: {:?}", e);
                continue;
            }

            let resp_buf = self.process_request(&req_buf);

            let written = if writer.available_bytes() >= resp_buf.len() {
                match writer.write_all(&resp_buf) {
                    Ok(_) => resp_buf.len(),
                    Err(e) => {
                        error!("gpu_nv: write error: {:?}", e);
                        0
                    }
                }
            } else {
                error!(
                    "gpu_nv: response buffer too small: have={} need={}",
                    writer.available_bytes(),
                    resp_buf.len()
                );
                0
            };

            if let Err(e) =
                self.control_queue
                    .lock()
                    .unwrap()
                    .add_used(&mem, head_index, written as u32)
            {
                error!("gpu_nv: add_used error: {:?}", e);
            }
            used_any = true;
        }

        used_any
    }
}
