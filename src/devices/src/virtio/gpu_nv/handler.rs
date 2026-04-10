// src/devices/src/virtio/gpu_nv/handler.rs
//
// Request processing — dispatches incoming virtqueue messages from the guest
// to the appropriate host-side handler.

use crate::virtio::gpu_nv::allowlist::AllowedIoctls;
use crate::virtio::gpu_nv::device::HostFd;
use crate::virtio::gpu_nv::worker::Worker;
use crate::virtio::gpu_nv::{
    bytes_of, ioc_nr, NvgpuIoctlReq, NvgpuIoctlResp, NvgpuMsgHdr, NvgpuOpenReq, NvgpuOpenResp,
    NVGPU_MSG_CLOSE, NVGPU_MSG_GET_PROC_FILES, NVGPU_MSG_GET_SYS_FILES, NVGPU_MSG_IOCTL,
    NVGPU_MSG_MMAP, NVGPU_MSG_MUNMAP, NVGPU_MSG_OPEN,
};
use std::os::unix::io::AsRawFd;

/// NVIDIA ioctl numbers that carry embedded pointers.
const NV_ESC_RM_CONTROL: u32 = 0x2a;
const NV_ESC_RM_ALLOC: u32 = 0x2b;

impl Worker {
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

        if std::env::var("NVGPU_LOG_IOCTLS").is_ok() {
            let msg_type = hdr.msg_type;
            let handle = hdr.handle;
            log::debug!(
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
            NVGPU_MSG_GET_PROC_FILES => self.handle_get_proc_files(),
            NVGPU_MSG_GET_SYS_FILES => self.handle_get_sys_files(),
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
            512.. => {
                // DRI device — index into config.dri_devices
                let idx = (req.device_type - 512) as usize;
                match self.config.dri_devices.get(idx) {
                    Some(dri) => format!("/dev/dri/{}", dri.name),
                    None => return self.error_response(0, -libc::ENODEV),
                }
            }
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

        log::error!(
            "virtio-gpu-nv: OPEN handle={} device_type={} path={}",
            handle,
            device_type,
            path
        );

        self.fd_table.insert(
            handle,
            HostFd {
                fd: file,
                device_type,
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
            log::debug!("virtio-gpu-nv: blocked ioctl nr=0x{:02x}", nr);
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
            log::error!(
                "virtio-gpu-nv: ioctl handle={} cmd=0x{:x} nr=0x{:x} \
                 data_len={} nested_len={}",
                handle,
                cmd,
                nr,
                data_len,
                nested_len
            );
        }

        // Route to simple or complex path.
        // In handle_ioctl, replace the match arm:
        match nr {
            NV_ESC_RM_CONTROL | NV_ESC_RM_ALLOC => self.execute_complex_ioctl(&req, data, nested),
            _ => {
                // Check fd-translation table before falling through to simple path.
                if let Some(entry) = AllowedIoctls::fd_translation(nr) {
                    self.execute_fd_translation_ioctl(&req, data, entry.payload_offset)
                } else {
                    self.execute_simple_ioctl(&req, data)
                }
            }
        }
    }

    /// Generic handler for ioctls that carry a guest fd at a known payload offset.
    /// Translates handle → host fd number, then forwards as a normal simple ioctl.
    fn execute_fd_translation_ioctl(
        &mut self,
        req: &NvgpuIoctlReq,
        data: &[u8],
        payload_offset: usize,
    ) -> Vec<u8> {
        // Bounds-check: we need 4 bytes at payload_offset.
        if data.len() < payload_offset + 4 {
            return self.error_response(req.hdr.handle, -libc::EINVAL);
        }

        // Guest driver placed the VMM handle here, not a raw fd number.
        let other_handle =
            u32::from_le_bytes(data[payload_offset..payload_offset + 4].try_into().unwrap());

        // Resolve handle → host fd number.
        let other_raw_fd = match self.fd_table.get(&other_handle) {
            Some(f) => f.fd.as_raw_fd(),
            None => {
                log::error!(
                    "virtio-gpu-nv: fd-translation ioctl nr=0x{:x}: \
                 unknown handle {}",
                    ioc_nr(req.cmd),
                    other_handle
                );
                return self.error_response(req.hdr.handle, -libc::EBADF);
            }
        };

        // Build a patched copy of the data with the real host fd in place.
        let mut buf = data.to_vec();
        buf[payload_offset..payload_offset + 4]
            .copy_from_slice(&(other_raw_fd as u32).to_le_bytes());

        if std::env::var("NVGPU_LOG_IOCTLS").is_ok() {
            let handle = req.hdr.handle;
            log::error!(
                "virtio-gpu-nv: fd-translation nr=0x{:x} handle={} \
             other_handle={} -> host_fd={}",
                ioc_nr(req.cmd),
                handle,
                other_handle,
                other_raw_fd
            );
        }

        // Now it's just a normal simple ioctl with the patched buffer.
        self.execute_simple_ioctl_with_buf(req, buf)
    }

    // ── Simple ioctl: flat struct, no embedded pointers ──────────────────────

    /// Issue the ioctl with a pre-built mutable buffer and return the response.
    fn execute_simple_ioctl_with_buf(&mut self, req: &NvgpuIoctlReq, mut buf: Vec<u8>) -> Vec<u8> {
        let handle = req.hdr.handle;
        let host_fd = match self.fd_table.get(&handle) {
            Some(f) => f,
            None => return self.error_response(handle, -libc::EBADF),
        };

        let ret = unsafe {
            libc::ioctl(
                host_fd.fd.as_raw_fd(),
                req.cmd as libc::c_ulong,
                buf.as_mut_ptr() as *mut libc::c_void,
            )
        };

        let errno = if ret < 0 { errno_val() } else { 0 };

        let nr = ioc_nr(req.cmd);
        if nr == 0xd7 || nr == 0xd6 {
            log::error!(
                "virtio-gpu-nv: nr=0x{:02x} ret={} errno={} first16={:02x?}",
                nr,
                ret,
                errno,
                &buf[..buf.len().min(16)]
            );
        }
        if std::env::var("NVGPU_LOG_IOCTLS").is_ok() && ioc_nr(req.cmd) == 0xc9 {
            log::error!(
                "virtio-gpu-nv: CARD_INFO/REGISTER_FD nr=0xc9 \
             ret={} errno={} buf={:02x?}",
                ret,
                errno,
                &buf[..buf.len().min(16)]
            );
        }

        let (status, out_len) = if ret < 0 {
            (-errno, 0u32)
        } else {
            (ret as i32, buf.len() as u32)
        };

        let resp_hdr = NvgpuIoctlResp {
            hdr: NvgpuMsgHdr {
                msg_type: NVGPU_MSG_IOCTL,
                handle,
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

    fn execute_simple_ioctl(&mut self, req: &NvgpuIoctlReq, data: &[u8]) -> Vec<u8> {
        self.execute_simple_ioctl_with_buf(req, data.to_vec())
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
        const RIGHTS_OFFSET: usize = 24; // pRightsRequested in NVOS64 only

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

        // After patching pAllocParms/params pointer, also zero pRightsRequested
        // for RM_ALLOC (NVOS64). For RM_CONTROL (NVOS54) offset 24 is paramsSize
        // which must not be zeroed — distinguish by ioc_nr.
        if ioc_nr(req.cmd) == NV_ESC_RM_ALLOC {
            if top.len() >= RIGHTS_OFFSET + PTR_SIZE {
                // pRightsRequested — null it out; we never pass access masks
                top[RIGHTS_OFFSET..RIGHTS_OFFSET + PTR_SIZE]
                    .copy_from_slice(&0u64.to_le_bytes());
            }
        }

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
    // GET_PROC_FILES — send entire /proc/driver/nvidia tree to guest
    // ─────────────────────────────────────────────────────────────────────────

    fn handle_get_proc_files(&self) -> Vec<u8> {
        let mut payload: Vec<u8> = Vec::new();

        for f in &self.config.extra_proc {
            let path = f.guest_path.as_bytes();
            let content = f.content.as_bytes();
            payload.extend_from_slice(&(path.len() as u32).to_le_bytes());
            payload.extend_from_slice(&(content.len() as u32).to_le_bytes());
            payload.extend_from_slice(path);
            payload.extend_from_slice(content);
        }

        // Terminator
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());

        log::debug!(
            "virtio-gpu-nv: GET_PROC_FILES {} files {} bytes",
            self.config.extra_proc.len(),
            payload.len()
        );
        payload
    }

    // ─────────────────────────────────────────────────────────────────────────
    // GET_SYS_FILES — send host sysfs content + DRI device list to guest
    //
    // Response format:
    //   Section 1 — sysfs files (same streaming format as GET_PROC_FILES):
    //     [path_len:u32][content_len:u32][path bytes][content bytes] ...
    //     terminated by [0u32][0u32]
    //
    //   Section 2 — DRI devices:
    //     [num_dri:u32]
    //     per device: [name_len:u32][major:u32][minor:u32][name bytes]
    // ─────────────────────────────────────────────────────────────────────────

    fn handle_get_sys_files(&self) -> Vec<u8> {
        let mut payload: Vec<u8> = Vec::new();

        // ── Section 1: sysfs files ───────────────────────────────────────────
        for f in &self.config.sys_files {
            let path = f.path.as_bytes();
            let content = &f.content;
            payload.extend_from_slice(&(path.len() as u32).to_le_bytes());
            payload.extend_from_slice(&(content.len() as u32).to_le_bytes());
            payload.extend_from_slice(path);
            payload.extend_from_slice(content);
        }
        // Terminator
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());

        // ── Section 2: DRI device nodes ──────────────────────────────────────
        // Per-device wire format:
        //   [name_len:u32][major:u32][minor:u32][gpu_id:u32][name bytes]
        payload.extend_from_slice(&(self.config.dri_devices.len() as u32).to_le_bytes());
        for dev in &self.config.dri_devices {
            let name = dev.name.as_bytes();
            payload.extend_from_slice(&(name.len() as u32).to_le_bytes());
            payload.extend_from_slice(&dev.major.to_le_bytes());
            payload.extend_from_slice(&dev.minor.to_le_bytes());
            payload.extend_from_slice(&dev.gpu_id.to_le_bytes());
            payload.extend_from_slice(name);
        }

        log::debug!(
            "virtio-gpu-nv: GET_SYS_FILES {} sys files, {} DRI devices, {} bytes",
            self.config.sys_files.len(),
            self.config.dri_devices.len(),
            payload.len()
        );

        payload
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
