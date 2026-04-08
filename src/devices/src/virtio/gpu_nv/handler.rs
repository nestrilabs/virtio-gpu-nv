// src/devices/src/virtio/gpu_nv/handler.rs
//
// Request processing — dispatches incoming virtqueue messages from the guest
// to the appropriate host-side handler.

use std::os::unix::io::AsRawFd;

use crate::virtio::gpu_nv::device::HostFd;
use crate::virtio::gpu_nv::{
    bytes_of, ioc_nr, GpuNv, NvgpuIoctlReq, NvgpuIoctlResp, NvgpuMsgHdr, NvgpuOpenReq,
    NvgpuOpenResp, NVGPU_MSG_CLOSE, NVGPU_MSG_IOCTL, NVGPU_MSG_MMAP, NVGPU_MSG_MUNMAP,
    NVGPU_MSG_OPEN,
};

/// NVIDIA ioctl numbers that carry embedded pointers.
const NV_ESC_RM_CONTROL: u32 = 0x2a;
const NV_ESC_RM_ALLOC: u32 = 0x2b;

impl GpuNv {
    // ─────────────────────────────────────────────────────────────────────────
    // Top-level dispatcher
    // ─────────────────────────────────────────────────────────────────────────

    /// Decode one virtqueue request buffer and return the serialised response.
    pub fn process_request(&mut self, req_buf: &[u8]) -> Vec<u8> {
        if req_buf.len() < std::mem::size_of::<NvgpuMsgHdr>() {
            return self.error_response(0, -libc::EINVAL);
        }

        // SAFETY: req_buf is at least sizeof(NvgpuMsgHdr) bytes long and we
        // only read the first field here.
        let hdr: NvgpuMsgHdr =
            unsafe { std::ptr::read_unaligned(req_buf.as_ptr() as *const NvgpuMsgHdr) };

        let msg_type = hdr.msg_type;
        let handle = hdr.handle;

        if std::env::var("NVGPU_LOG_IOCTLS").is_ok() {
            eprintln!(
                "virtio-gpu-nv: msg_type={} handle={} buf_len={}",
                msg_type,
                handle,
                req_buf.len()
            );
        }

        match hdr.msg_type {
            NVGPU_MSG_OPEN => self.handle_open(req_buf),
            NVGPU_MSG_CLOSE => self.handle_close(&hdr),
            NVGPU_MSG_IOCTL => self.handle_ioctl(req_buf),
            NVGPU_MSG_MMAP => self.handle_mmap(req_buf),
            NVGPU_MSG_MUNMAP => self.handle_munmap(req_buf),
            _ => self.error_response(hdr.handle, -libc::ENOSYS),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // OPEN — open a new host /dev/nvidia* on behalf of the guest
    // ─────────────────────────────────────────────────────────────────────────

    fn handle_open(&mut self, req_buf: &[u8]) -> Vec<u8> {
        if req_buf.len() < std::mem::size_of::<NvgpuOpenReq>() {
            return self.error_response(0, -libc::EINVAL);
        }

        let req: NvgpuOpenReq =
            unsafe { std::ptr::read_unaligned(req_buf.as_ptr() as *const NvgpuOpenReq) };

        let device_type = req.device_type;

        // Map device_type → host path
        let path = match req.device_type {
            0..=247 => format!("/dev/nvidia{}", device_type),
            255 => "/dev/nvidiactl".to_string(),
            256 => "/dev/nvidia-uvm".to_string(),
            257 => "/dev/nvidia-uvm-tools".to_string(),
            258 => "/dev/nvidia-modeset".to_string(),
            _ => return self.error_response(0, -libc::EINVAL),
        };

        // Each guest open() gets a fresh host FD so the NVIDIA driver
        // assigns a new RM client with isolated channels + allocations.
        let file = match std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
        {
            Ok(f) => f,
            Err(e) => {
                return self.error_response(0, -e.raw_os_error().unwrap_or(libc::EIO));
            }
        };

        let handle = self.next_handle;
        self.next_handle += 1;

        self.fd_table.insert(
            handle,
            HostFd {
                fd: file,
                device_type: req.device_type,
                mapping_ids: Vec::new(),
            },
        );

        let resp = NvgpuOpenResp {
            hdr: NvgpuMsgHdr {
                msg_type: NVGPU_MSG_OPEN,
                handle,
                status: 0,
                padding: 0,
            },
        };

        bytes_of(&resp)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CLOSE — release host FD and all its GPU mappings
    // ─────────────────────────────────────────────────────────────────────────

    pub(crate) fn handle_close(&mut self, hdr: &NvgpuMsgHdr) -> Vec<u8> {
        let handle = hdr.handle;
        if let Some(host_fd) = self.fd_table.remove(&handle) {
            // Clean up every KVM memory slot this FD owns before dropping fd.
            let ids: Vec<u32> = host_fd.mapping_ids.clone();
            for mapping_id in ids {
                self.destroy_mapping(mapping_id);
            }
            // host_fd.fd is dropped here → host FD closed.
        }

        bytes_of(&NvgpuMsgHdr {
            msg_type: NVGPU_MSG_CLOSE,
            handle: hdr.handle,
            status: 0,
            padding: 0,
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // IOCTL — forward to host driver
    // ─────────────────────────────────────────────────────────────────────────

    fn handle_ioctl(&mut self, req_buf: &[u8]) -> Vec<u8> {
        let hdr_size = std::mem::size_of::<NvgpuIoctlReq>();
        if req_buf.len() < hdr_size {
            return self.error_response(0, -libc::EINVAL);
        }

        let req: NvgpuIoctlReq =
            unsafe { std::ptr::read_unaligned(req_buf.as_ptr() as *const NvgpuIoctlReq) };

        // Security: reject unknown ioctl commands.
        let nr = ioc_nr(req.cmd);
        if !self.allowed_ioctls.is_allowed(nr) {
            eprintln!("virtio-gpu-nv: blocked ioctl nr=0x{:02x}", nr);
            return self.error_response(req.hdr.handle, -libc::ENOTTY);
        }

        // Bounds-check payload lengths.
        let data_end = hdr_size + req.data_len as usize;
        let nested_end = hdr_size + req.nested_offset as usize + req.nested_len as usize;

        if data_end > req_buf.len() || nested_end > req_buf.len() {
            return self.error_response(req.hdr.handle, -libc::EINVAL);
        }

        let data = &req_buf[hdr_size..data_end];
        let nested = if req.nested_len > 0 {
            let start = hdr_size + req.nested_offset as usize;
            &req_buf[start..start + req.nested_len as usize]
        } else {
            &[]
        };

        let handle = req.hdr.handle;
        let cmd = req.cmd;
        let data_len = req.data_len;
        let nested_len = req.nested_len;

        if std::env::var("NVGPU_LOG_IOCTLS").is_ok() {
            eprintln!(
                "virtio-gpu-nv: ioctl handle={} cmd=0x{:x} nr=0x{:x} \
                 data_len={} nested_len={}",
                handle, cmd, nr, data_len, nested_len
            );
        }

        // Route to simple or complex path.
        match nr {
            NV_ESC_RM_CONTROL | NV_ESC_RM_ALLOC => self.execute_complex_ioctl(&req, data, nested),
            _ => self.execute_simple_ioctl(&req, data),
        }
    }

    // ── Simple ioctl: flat struct, no embedded pointers ──────────────────────

    fn execute_simple_ioctl(&mut self, req: &NvgpuIoctlReq, data: &[u8]) -> Vec<u8> {
        let handle = req.hdr.handle;
        let host_fd = match self.fd_table.get(&handle) {
            Some(f) => f,
            None => return self.error_response(req.hdr.handle, -libc::EBADF),
        };

        // Copy data into a mutable buffer — the ioctl may write back in place.
        let mut buf = data.to_vec();

        let ret = unsafe {
            libc::ioctl(
                host_fd.fd.as_raw_fd(),
                req.cmd as libc::c_ulong,
                buf.as_mut_ptr() as *mut libc::c_void,
            )
        };

        let (status, out_len) = if ret < 0 {
            (-errno_val(), 0u32)
        } else {
            (ret as i32, buf.len() as u32)
        };

        let resp_hdr = NvgpuIoctlResp {
            hdr: NvgpuMsgHdr {
                msg_type: NVGPU_MSG_IOCTL,
                handle: req.hdr.handle,
                status,
                padding: 0,
            },
            data_len: out_len,
            nested_len: 0,
        };

        let mut out = bytes_of(&resp_hdr);
        if out_len > 0 {
            out.extend_from_slice(&buf);
        }
        out
    }

    // ── Complex ioctl: NV_ESC_RM_CONTROL / NV_ESC_RM_ALLOC ──────────────────
    //
    // These carry embedded guest pointers that we must:
    //   1. Replace with a host pointer to our nested_buf.
    //   2. Execute the ioctl on the host fd.
    //   3. Restore the original guest pointer value before sending back.

    fn execute_complex_ioctl(
        &mut self,
        req: &NvgpuIoctlReq,
        data: &[u8],
        nested: &[u8],
    ) -> Vec<u8> {
        let handle = req.hdr.handle;
        let host_fd = match self.fd_table.get(&handle) {
            Some(f) => f,
            None => return self.error_response(req.hdr.handle, -libc::EBADF),
        };

        let raw_fd = host_fd.fd.as_raw_fd();

        // Mutable copy of the top-level struct.
        let mut top = data.to_vec();
        // Mutable copy of the nested buffer — will be modified in place.
        let mut nested_buf = nested.to_vec();

        // Save original guest pointer bytes (to restore after the ioctl).
        // Both structs have the pointer at byte offset 16 (after 4×u32 fields).
        const PTR_OFFSET: usize = 16;
        const PTR_SIZE: usize = 8;

        if top.len() < PTR_OFFSET + PTR_SIZE {
            return self.error_response(req.hdr.handle, -libc::EINVAL);
        }

        let mut saved_guest_ptr = [0u8; PTR_SIZE];
        saved_guest_ptr.copy_from_slice(&top[PTR_OFFSET..PTR_OFFSET + PTR_SIZE]);

        // Patch the pointer field with the address of our host buffer.
        let host_ptr_val: u64 = if nested_buf.is_empty() {
            0
        } else {
            nested_buf.as_mut_ptr() as u64
        };
        top[PTR_OFFSET..PTR_OFFSET + PTR_SIZE].copy_from_slice(&host_ptr_val.to_ne_bytes());

        let ret = unsafe {
            libc::ioctl(
                raw_fd,
                req.cmd as libc::c_ulong,
                top.as_mut_ptr() as *mut libc::c_void,
            )
        };

        let status = if ret < 0 { -errno_val() } else { ret as i32 };

        // Restore the original guest pointer so the guest driver can use it
        // for copy_to_user.
        top[PTR_OFFSET..PTR_OFFSET + PTR_SIZE].copy_from_slice(&saved_guest_ptr);

        let resp_hdr = NvgpuIoctlResp {
            hdr: NvgpuMsgHdr {
                msg_type: NVGPU_MSG_IOCTL,
                handle: req.hdr.handle,
                status,
                padding: 0,
            },
            data_len: top.len() as u32,
            nested_len: nested_buf.len() as u32,
        };

        let mut out = bytes_of(&resp_hdr);
        out.extend_from_slice(&top);
        out.extend_from_slice(&nested_buf);
        out
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    pub(crate) fn error_response(&self, handle: u32, status: i32) -> Vec<u8> {
        bytes_of(&NvgpuMsgHdr {
            msg_type: 0,
            handle,
            status,
            padding: 0,
        })
    }
}

/// Read errno from the OS.
fn errno_val() -> i32 {
    std::io::Error::last_os_error()
        .raw_os_error()
        .unwrap_or(libc::EIO)
}
