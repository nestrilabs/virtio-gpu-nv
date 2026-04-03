// crates/device/src/nvidia.rs
// Core backend: message dispatch, open/close, Phase 2 ioctl forwarding,
// and graceful/ungraceful teardown.
//
// --- Phase 5 note: /dev/nvidia-drm and /dev/nvidia-modeset ---
//
// Phases 1-4 only require /dev/nvidiactl, /dev/nvidia#, and /dev/nvidia-uvm.
// This holds for headless compute/encode (CUDA, NVENC, Vulkan headless).
//
// Phase 5 (compositor integration) may require /dev/nvidia-drm IF the guest
// compositor uses GBM (gbm_create_device) for buffer allocation, or wants
// DRM/KMS for display timing.  Both GBM and the DRM master path call into
// /dev/nvidia-drm.  If this becomes necessary:
//   - Add NV_DEV_DRM and NV_DEV_MODESET variants to DeviceKind in
//     protocol/src/messages.rs.
//   - Add the corresponding open/ioctl paths in device_path() below.
//   - Prefer EGLStreams over GBM in the guest compositor if possible;
//     EGLStreams does not require /dev/nvidia-drm and avoids this dependency.
use std::ffi::CString;
use std::os::fd::{FromRawFd, OwnedFd, RawFd};

use protocol::messages::*;

use crate::error::{DeviceError, Result};
use crate::handle_table::HandleTable;
use crate::shm::{ShmAllocator, ZoneConfig};

// ============================================================
// Device path helpers
// ============================================================

const MAX_GPU: u8 = 8;

fn device_path(kind: u8, index: u8) -> Result<CString> {
    let path = match kind {
        k if k == DeviceKind::Ctl as u8 => "/dev/nvidiactl".to_string(),
        k if k == DeviceKind::Gpu as u8 => {
            if index >= MAX_GPU {
                return Err(DeviceError::GpuIndexOutOfRange(index));
            }
            format!("/dev/nvidia{}", index)
        }
        k if k == DeviceKind::Uvm as u8 => "/dev/nvidia-uvm".to_string(),
        k if k == DeviceKind::Modeset as u8 => "/dev/nvidia-modeset".to_string(),
        other => return Err(DeviceError::InvalidDeviceKind(other)),
    };
    Ok(CString::new(path).unwrap())
}

// ============================================================
// NvidiaBackend
// ============================================================

pub struct NvidiaBackend {
    handles: HandleTable,
    shm: ShmAllocator,
}
impl NvidiaBackend {
    /// Create a backend with a custom SHM zone config.
    pub fn new(cfg: ZoneConfig) -> Self {
        Self {
            handles: HandleTable::new(),
            shm: ShmAllocator::new(cfg),
        }
    }

    /// Create a backend with the default 256 MiB zone split.
    pub fn with_default_zones() -> Self {
        Self::new(ZoneConfig::default_256mib())
    }

    /// Total SHM BAR size (for VMM config space).
    pub fn shm_total_size(&self) -> u64 {
        self.shm.total_size()
    }

    /// Raw memfd fd (for KVM memslot creation).
    pub fn shm_memfd_raw(&self) -> i32 {
        self.shm.memfd_raw()
    }

    pub fn shm_base_ptr(&self) -> *mut u8 {
        self.shm.base_ptr()
    }

    /// Override the SHM base pointer to the guest memory HVA.
    /// Called by the VMM after guest memory setup.
    pub fn set_shm_base(&mut self, ptr: *mut u8) {
        self.shm.set_base_ptr(ptr);
    }

    /// Create a minimal backend suitable for unit tests (8-page total BAR).
    #[cfg(test)]
    pub fn for_test() -> Self {
        Self::new(ZoneConfig {
            uc_size: 4096 * 2,
            wc_size: 4096 * 4,
            wb_size: 4096 * 2,
        })
    }

    // ------------------------------------------------------------------
    // Teardown
    //
    // Called by the VMM on:
    //   - normal VM shutdown (virtio device reset before exit)
    //   - ungraceful VM exit (SIGKILL, crash, libkrun teardown)
    //
    // Draining the handle table closes every host fd, which triggers the
    // host NVIDIA driver's fd-release path and frees all RM objects.
    // Analogous to nvproxy's Release() in nvproxy.go.
    // ------------------------------------------------------------------

    pub fn teardown(&mut self) {
        log::info!(
            "NvidiaBackend::teardown: draining {} handles",
            self.handles.len()
        );
        self.handles.drain_all();
    }

    // ------------------------------------------------------------------
    // Top-level dispatch
    // ------------------------------------------------------------------

    pub fn dispatch(&mut self, req_buf: &[u8], resp_buf: &mut [u8]) -> usize {
        if req_buf.len() < size_of::<MsgHeader>() {
            return self.write_error_resp(resp_buf, Status::InvalidMsgType, 0, 0);
        }
        let hdr = read_struct::<MsgHeader>(req_buf, 0);
        let cookie = hdr.cookie;

        let msg_type = match hdr.msg_type {
            t if t == MsgType::Open as u32 => MsgType::Open,
            t if t == MsgType::Close as u32 => MsgType::Close,
            t if t == MsgType::Ioctl as u32 => MsgType::Ioctl,
            other => {
                log::warn!("unknown msg_type {}", other);
                return self.write_error_resp(resp_buf, Status::InvalidMsgType, cookie, 0);
            }
        };

        let payload = &req_buf[size_of::<MsgHeader>()..];
        match msg_type {
            MsgType::Open => self.handle_open(cookie, payload, resp_buf),
            MsgType::Close => self.handle_close(cookie, payload, resp_buf),
            MsgType::Ioctl => self.handle_ioctl(cookie, payload, resp_buf),
        }
    }

    // ------------------------------------------------------------------
    // OPEN
    // ------------------------------------------------------------------

    fn handle_open(&mut self, cookie: u64, payload: &[u8], resp_buf: &mut [u8]) -> usize {
        if payload.len() < size_of::<OpenReq>() {
            return self.write_error_resp(resp_buf, Status::InvalidDevice, cookie, 0);
        }
        let req = read_struct::<OpenReq>(payload, 0);

        let path = match device_path(req.kind, req.index) {
            Ok(p) => p,
            Err(e) => {
                log::warn!("handle_open: {}", e);
                return self.write_error_resp(resp_buf, Status::InvalidDevice, cookie, 0);
            }
        };

        let raw_fd = unsafe { libc::open(path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC) };

        if raw_fd < 0 {
            let err = std::io::Error::last_os_error();
            let errno = err.raw_os_error().unwrap_or(0);
            log::warn!("open({:?}) failed: {}", path, err);
            return self.write_error_resp(resp_buf, Status::OpenFailed, cookie, errno);
        }

        let guest_handle = self.handles.insert(unsafe { OwnedFd::from_raw_fd(raw_fd) });
        log::debug!("open {:?} → handle={}", path, guest_handle);

        write_ok(resp_buf, cookie, &OpenResp { guest_handle })
    }

    // ------------------------------------------------------------------
    // CLOSE
    // ------------------------------------------------------------------

    fn handle_close(&mut self, cookie: u64, payload: &[u8], resp_buf: &mut [u8]) -> usize {
        if payload.len() < size_of::<CloseReq>() {
            return self.write_error_resp(resp_buf, Status::BadHandle, cookie, 0);
        }
        let req = read_struct::<CloseReq>(payload, 0);

        match self.handles.remove(req.guest_handle) {
            Ok(()) => {
                log::debug!("close handle={}", req.guest_handle);
                write_ok(resp_buf, cookie, &CloseResp { _pad: 0 })
            }
            Err(_) => self.write_error_resp(resp_buf, Status::BadHandle, cookie, 0),
        }
    }

    // ------------------------------------------------------------------
    // IOCTL — top-level
    // ------------------------------------------------------------------

    fn handle_ioctl(&mut self, cookie: u64, payload: &[u8], resp_buf: &mut [u8]) -> usize {
        if payload.len() < size_of::<IoctlReq>() {
            return self.write_error_resp(resp_buf, Status::InvalidMsgType, cookie, 0);
        }
        let ireq = read_struct::<IoctlReq>(payload, 0);
        let param_in =
            &payload[size_of::<IoctlReq>()..size_of::<IoctlReq>() + ireq.param_size as usize];

        let host_fd = match self.handles.get_raw(ireq.guest_handle) {
            Ok(fd) => fd,
            Err(_) => return self.write_error_resp(resp_buf, Status::BadHandle, cookie, 0),
        };

        let escape = (ireq.request & 0xFF) as u32;

        log::trace!(
            "IOCTL: handle={} escape=0x{:02x} req=0x{:x} param_size={}",
            ireq.guest_handle,
            escape,
            ireq.request,
            ireq.param_size
        );

        use abi::ioctl::*;
        match escape {
            // ---------------------------------------------------------------
            // FD-carrying ioctls — need handle translation
            // ---------------------------------------------------------------
            NV_ESC_REGISTER_FD | NV_ESC_ALLOC_OS_EVENT | NV_ESC_FREE_OS_EVENT => {
                self.dispatch_fd_carrying(cookie, host_fd, ireq.request, escape, param_in, resp_buf)
            }

            // ---------------------------------------------------------------
            // Map memory — needs FD translation + SHM allocation
            // ---------------------------------------------------------------
            NV_ESC_RM_MAP_MEMORY => {
                self.dispatch_map_memory(cookie, host_fd, ireq.request, param_in, resp_buf)
            }

            // ---------------------------------------------------------------
            // RM control requires nested handling
            // ---------------------------------------------------------------
            NV_ESC_RM_CONTROL => self.dispatch_nested(
                cookie,
                host_fd,
                ireq.request,
                param_in,
                resp_buf,
                32,
                16,
                24,
            ),

            // ---------------------------------------------------------------
            // RM alloc as well..
            // ---------------------------------------------------------------
            NV_ESC_RM_ALLOC | NV_ESC_RM_ALLOC_MEMORY => self.dispatch_nested(
                cookie,
                host_fd,
                ireq.request,
                param_in,
                resp_buf,
                48,
                16,
                32,
            ),

            // ---------------------------------------------------------------
            // Everything else — simple passthrough to host
            // ---------------------------------------------------------------
            _other => {
                // Catch dedicated EXPORT/IMPORT escapes (0x5C, 0x5D) if the
                // library uses them instead of (or in addition to) RM_CONTROL.
                if _other == 0x5C || _other == 0x5D {
                    log::info!(
                        ">>> DEDICATED EXPORT/IMPORT escape=0x{:02x} param_size={} param_in={:02x?}",
                        _other,
                        param_in.len(),
                        &param_in[..std::cmp::min(param_in.len(), 64)]
                    );
                }
                log::debug!(
                    "ioctl passthrough escape=0x{:02x} size={}",
                    _other,
                    param_in.len()
                );
                self.dispatch_simple(cookie, host_fd, ireq.request, param_in, resp_buf)
            }
        }
    }

    // ------------------------------------------------------------------
    // Nested-pointer ioctl (RM_CONTROL, RM_ALLOC..)
    // ------------------------------------------------------------------

    fn dispatch_nested(
        &self,
        cookie: u64,
        host_fd: RawFd,
        request: u64,
        param_in: &[u8],
        resp_buf: &mut [u8],
        outer_size: usize,
        ptr_offset: usize,
        _size_offset: usize,
    ) -> usize {
        if param_in.len() < outer_size {
            return self.write_error_resp(resp_buf, Status::IoctlFailed, cookie, libc::EINVAL);
        }

        let mut outer = param_in[..outer_size].to_vec();
        let nested_in = &param_in[outer_size..];

        let escape = (request & 0xFF) as u32;

        if !nested_in.is_empty() {
            // Guest sent nested params — allocate host buffer, point struct at it
            let nested_size = nested_in.len();
            let mut host_buf = vec![0u8; nested_size];
            host_buf.copy_from_slice(nested_in);

            // ---------------------------------------------------------------
            // Translate guest_handle → host fd for fd-carrying RM_CONTROLs
            //
            // The guest driver already translated the raw guest fd to a
            // guest_handle. We now translate that handle to a real host fd
            // so the host kernel can resolve it.
            // ---------------------------------------------------------------
            let mut saved_nested_handle: Option<(usize, i32)> = None; // (offset, guest_handle_as_i32)

            if escape == 0x2A && outer.len() >= 12 {
                let cmd = u32::from_le_bytes(outer[8..12].try_into().unwrap());

                if cmd == 0x3d05 && host_buf.len() >= 20 {
                    // EXPORT_OBJECT_TO_FD: guest_handle at offset 16 in nested
                    let guest_handle_val = i32::from_le_bytes(host_buf[16..20].try_into().unwrap());

                    match self.handles.get_raw(guest_handle_val as u64) {
                        Ok(real_fd) => {
                            log::debug!(
                                "EXPORT_TO_FD: handle {} → host fd {}",
                                guest_handle_val,
                                real_fd
                            );
                            saved_nested_handle = Some((16, guest_handle_val));
                            host_buf[16..20].copy_from_slice(&(real_fd as i32).to_le_bytes());
                        }
                        Err(_) => {
                            log::warn!("EXPORT_TO_FD: bad guest_handle {}", guest_handle_val);
                            return self.write_error_resp(resp_buf, Status::BadHandle, cookie, 0);
                        }
                    }
                }

                if cmd == 0x3d06 && host_buf.len() >= 4 {
                    // IMPORT_OBJECT_FROM_FD: guest_handle at offset 0 in nested
                    let guest_handle_val = i32::from_le_bytes(host_buf[0..4].try_into().unwrap());

                    match self.handles.get_raw(guest_handle_val as u64) {
                        Ok(real_fd) => {
                            log::debug!(
                                "IMPORT_FROM_FD: handle {} → host fd {}",
                                guest_handle_val,
                                real_fd
                            );
                            saved_nested_handle = Some((0, guest_handle_val));
                            host_buf[0..4].copy_from_slice(&(real_fd as i32).to_le_bytes());
                        }
                        Err(_) => {
                            log::warn!("IMPORT_FROM_FD: bad guest_handle {}", guest_handle_val);
                            return self.write_error_resp(resp_buf, Status::BadHandle, cookie, 0);
                        }
                    }
                }
            }

            // Set pointer in outer struct to host buffer address
            let host_ptr = host_buf.as_mut_ptr() as u64;
            outer[ptr_offset..ptr_offset + 8].copy_from_slice(&host_ptr.to_le_bytes());

            // Call host ioctl — paramsSize field is untouched (may be 0)
            let rc = unsafe { libc::ioctl(host_fd, request as libc::Ioctl, outer.as_mut_ptr()) };
            if rc < 0 {
                let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
                log::warn!("nested ioctl(0x{:x}) failed: errno={}", request, errno);
                return self.write_error_resp(resp_buf, Status::IoctlFailed, cookie, errno);
            }

            // Log RM status
            if escape == 0x2a {
                let status = u32::from_le_bytes(outer[28..32].try_into().unwrap());
                let cmd = u32::from_le_bytes(outer[8..12].try_into().unwrap());
                log::debug!("(if) RM_CONTROL cmd=0x{:08x} status=0x{:x}", cmd, status);
            } else if escape == 0x2b {
                let status = u32::from_le_bytes(outer[40..44].try_into().unwrap());
                let hclass = u32::from_le_bytes(outer[12..16].try_into().unwrap());
                log::debug!(
                    "(if) RM_ALLOC hClass=0x{:04x} status=0x{:x}",
                    hclass,
                    status
                );
            }

            // Restore guest_handle in host_buf before sending back to guest
            if let Some((offset, handle_val)) = saved_nested_handle {
                host_buf[offset..offset + 4].copy_from_slice(&handle_val.to_le_bytes());
            }

            // Zero pointer before sending back to guest
            outer[ptr_offset..ptr_offset + 8].copy_from_slice(&0u64.to_le_bytes());

            // Build response: outer + updated nested params
            let mut combined = outer;
            combined.extend_from_slice(&host_buf);
            self.write_ioctl_resp(resp_buf, cookie, &combined)
        } else {
            // No nested params — straightforward passthrough
            let rc = unsafe { libc::ioctl(host_fd, request as libc::Ioctl, outer.as_mut_ptr()) };
            if rc < 0 {
                let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
                log::warn!(
                    "nested ioctl(0x{:x}) no-params failed: errno={}",
                    request,
                    errno
                );
                return self.write_error_resp(resp_buf, Status::IoctlFailed, cookie, errno);
            }

            if escape == 0x2a {
                let status = u32::from_le_bytes(outer[28..32].try_into().unwrap());
                let cmd = u32::from_le_bytes(outer[8..12].try_into().unwrap());
                log::debug!("(else) RM_CONTROL cmd=0x{:08x} status=0x{:x}", cmd, status);
            } else if escape == 0x2b {
                let status = u32::from_le_bytes(outer[40..44].try_into().unwrap());
                let hclass = u32::from_le_bytes(outer[12..16].try_into().unwrap());
                log::debug!(
                    "(else) RM_ALLOC hClass=0x{:04x} status=0x{:x}",
                    hclass,
                    status
                );
            }

            // Zero pointer field in case host wrote something there
            outer[ptr_offset..ptr_offset + 8].copy_from_slice(&0u64.to_le_bytes());
            self.write_ioctl_resp(resp_buf, cookie, &outer)
        }
    }

    // ------------------------------------------------------------------
    // Simple ioctl
    // ------------------------------------------------------------------

    fn dispatch_simple(
        &self,
        cookie: u64,
        host_fd: RawFd,
        request: u64,
        param_in: &[u8],
        resp_buf: &mut [u8],
    ) -> usize {
        let mut param_buf = param_in.to_vec();
        let rc = unsafe { libc::ioctl(host_fd, request as libc::Ioctl, param_buf.as_mut_ptr()) };
        if rc < 0 {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            log::warn!("ioctl(0x{:x}) failed: errno={}", request, errno);
            return self.write_error_resp(resp_buf, Status::IoctlFailed, cookie, errno);
        } else {
            let escape = (request & 0xFF) as u32;
            if escape == 0xd4 || escape == 0x4a {
                log::info!(
                    "dispatch_simple: escape=0x{:02x} succeeded, response={:02x?}",
                    escape,
                    &param_buf[..std::cmp::min(param_buf.len(), 64)]
                );
            }
        }
        self.write_ioctl_resp(resp_buf, cookie, &param_buf)
    }

    // ------------------------------------------------------------------
    // FD-carrying ioctl
    // ------------------------------------------------------------------

    fn dispatch_fd_carrying(
        &self,
        cookie: u64,
        host_fd: RawFd,
        request: u64,
        escape: u32,
        param_in: &[u8],
        resp_buf: &mut [u8],
    ) -> usize {
        use abi::ioctl::*;

        let fd_offset: usize = match escape {
            // nv_ioctl_register_fd_t: ctl_fd is the only field, offset 0.
            NV_ESC_REGISTER_FD => 0,
            // nv_ioctl_alloc_os_event_t: hClient(4) + hDevice(4) + fd @ offset 8
            NV_ESC_ALLOC_OS_EVENT => 8,
            // nv_ioctl_free_os_event_t: same layout as alloc, fd @ offset 8
            NV_ESC_FREE_OS_EVENT => 8,
            _ => return self.write_error_resp(resp_buf, Status::IoctlFailed, cookie, libc::ENOTTY),
        };

        if param_in.len() < fd_offset + 4 {
            return self.write_error_resp(resp_buf, Status::IoctlFailed, cookie, libc::EINVAL);
        }

        let guest_embedded: u64 = {
            let mut b = [0u8; 4];
            b.copy_from_slice(&param_in[fd_offset..fd_offset + 4]);
            u32::from_le_bytes(b) as u64
        };

        let host_embedded = match self.handles.get_raw(guest_embedded) {
            Ok(fd) => fd,
            Err(_) => {
                log::warn!("fd-carrying ioctl: bad embedded handle {}", guest_embedded);
                return self.write_error_resp(resp_buf, Status::BadHandle, cookie, 0);
            }
        };

        let mut param_buf = param_in.to_vec();
        param_buf[fd_offset..fd_offset + 4].copy_from_slice(&(host_embedded as i32).to_le_bytes());

        let rc = unsafe { libc::ioctl(host_fd, request as libc::Ioctl, param_buf.as_mut_ptr()) };

        if rc < 0 {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            log::warn!("fd-carrying ioctl(0x{:x}) failed: errno={}", request, errno);
            return self.write_error_resp(resp_buf, Status::IoctlFailed, cookie, errno);
        }

        // Restore guest handle in response so user-mode code reading it back
        // gets what it originally wrote.
        param_buf[fd_offset..fd_offset + 4].copy_from_slice(&(guest_embedded as i32).to_le_bytes());

        self.write_ioctl_resp(resp_buf, cookie, &param_buf)
    }

    // ------------------------------------------------------------------
    // NV_ESC_RM_MAP_MEMORY handler
    //
    // Wire layout of param_in (IoctlNVOS33ParametersWithFD):
    //
    //   offset  0: NVOS33_PARAMETERS (48 bytes)
    //     offset  0: hClient       u32
    //     offset  4: hDevice       u32
    //     offset  8: hMemory       u32
    //     offset 12: pad           [4]u8
    //     offset 16: offset        u64
    //     offset 24: length        u64
    //     offset 32: pLinearAddress u64
    //     offset 40: status        u32
    //     offset 44: flags         u32
    //   offset 48: fd              i32   (guest handle → host fd)
    //   offset 52: pad             [4]u8
    //
    // Total: 56 bytes.
    //
    // Flow (ported from gVisor nvproxy frontend.go:rmMapMemory):
    //   1. Translate the embedded FD (guest handle → host fd).
    //   2. Call the host ioctl.
    //   3. If successful, read the updated flags to determine caching type.
    //   4. Allocate a region from the SHM BAR (correct zone per pgprot).
    //   5. mmap the host fd into the SHM region.
    //   6. Return the SHM offset, length, and pgprot to the guest.
    //   7. Restore the guest handle in the response params.
    // ------------------------------------------------------------------
    fn dispatch_map_memory(
        &mut self,
        cookie: u64,
        host_fd: RawFd,
        request: u64,
        param_in: &[u8],
        resp_buf: &mut [u8],
    ) -> usize {
        use crate::shm::PgprotKind;

        const _NVOS33_SIZE: usize = 48;
        const WITH_FD_SIZE: usize = 56;
        const FD_OFFSET: usize = 48;
        const LENGTH_OFFSET: usize = 24;
        const STATUS_OFFSET: usize = 40;
        const FLAGS_OFFSET: usize = 44;

        const FLAGS_CACHING_TYPE_SHIFT: u32 = 23;
        const FLAGS_CACHING_TYPE_MASK: u32 = 0x7;
        const CACHING_TYPE_CACHED: u32 = 0;
        const CACHING_TYPE_UNCACHED: u32 = 1;
        const CACHING_TYPE_WRITECOMBINED: u32 = 2;
        const CACHING_TYPE_WRITEBACK: u32 = 5;
        const CACHING_TYPE_DEFAULT: u32 = 6;
        const CACHING_TYPE_UNCACHED_WEAK: u32 = 7;

        if param_in.len() < WITH_FD_SIZE {
            return self.write_error_resp(resp_buf, Status::IoctlFailed, cookie, libc::EINVAL);
        }

        // --- Step 1: Translate embedded FD (guest handle → host fd) ---

        let guest_fd_handle = {
            let mut b = [0u8; 4];
            b.copy_from_slice(&param_in[FD_OFFSET..FD_OFFSET + 4]);
            i32::from_le_bytes(b) as u64
        };

        let host_map_fd = match self.handles.get_raw(guest_fd_handle) {
            Ok(fd) => fd,
            Err(_) => {
                log::warn!(
                    "NV_ESC_RM_MAP_MEMORY: bad embedded FD handle {}",
                    guest_fd_handle
                );
                return self.write_error_resp(resp_buf, Status::BadHandle, cookie, 0);
            }
        };

        let mut param_buf = param_in.to_vec();
        param_buf[FD_OFFSET..FD_OFFSET + 4].copy_from_slice(&(host_map_fd as i32).to_le_bytes());

        // --- Step 2: Call host ioctl ---

        let rc = unsafe { libc::ioctl(host_fd, request as libc::Ioctl, param_buf.as_mut_ptr()) };

        if rc < 0 {
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            log::warn!("NV_ESC_RM_MAP_MEMORY: host ioctl failed: errno={}", errno);
            // Restore guest handle before returning
            param_buf[FD_OFFSET..FD_OFFSET + 4]
                .copy_from_slice(&(guest_fd_handle as i32).to_le_bytes());
            return self.write_error_resp(resp_buf, Status::IoctlFailed, cookie, errno);
        }

        // --- Step 3: Check RM status and read updated fields ---

        let rm_status = u32::from_le_bytes(
            param_buf[STATUS_OFFSET..STATUS_OFFSET + 4]
                .try_into()
                .unwrap(),
        );

        // Restore guest handle in param_buf for copy-out regardless of status.
        param_buf[FD_OFFSET..FD_OFFSET + 4]
            .copy_from_slice(&(guest_fd_handle as i32).to_le_bytes());

        if rm_status != 0 {
            // RM returned an error status (NV_OK == 0).
            // Forward the params back so the guest can read the status field.
            log::debug!("NV_ESC_RM_MAP_MEMORY: RM status 0x{:x}", rm_status);
            return self.write_ioctl_resp(resp_buf, cookie, &param_buf);
        }

        let length = u64::from_le_bytes(
            param_buf[LENGTH_OFFSET..LENGTH_OFFSET + 8]
                .try_into()
                .unwrap(),
        );

        let flags = u32::from_le_bytes(
            param_buf[FLAGS_OFFSET..FLAGS_OFFSET + 4]
                .try_into()
                .unwrap(),
        );

        // --- Step 4: Determine pgprot from caching type ---
        //
        // The host driver may have updated the caching type in flags after
        // the ioctl (see nvproxy's rmMapMemory comment about this).

        let caching_type = (flags >> FLAGS_CACHING_TYPE_SHIFT) & FLAGS_CACHING_TYPE_MASK;

        let pgprot = match caching_type {
            CACHING_TYPE_CACHED | CACHING_TYPE_WRITEBACK => PgprotKind::WriteBack,
            CACHING_TYPE_WRITECOMBINED | CACHING_TYPE_DEFAULT => PgprotKind::WriteCombine,
            CACHING_TYPE_UNCACHED | CACHING_TYPE_UNCACHED_WEAK => PgprotKind::Uncached,
            other => {
                log::warn!(
                    "NV_ESC_RM_MAP_MEMORY: unknown caching type {}, defaulting to UC",
                    other
                );
                PgprotKind::Uncached
            }
        };

        // --- Step 5: Allocate SHM region ---

        let region = match self.shm.alloc(length, pgprot) {
            Ok(r) => r,
            Err(e) => {
                log::error!("NV_ESC_RM_MAP_MEMORY: SHM alloc failed: {}", e);
                return self.write_error_resp(resp_buf, Status::IoctlFailed, cookie, libc::ENOMEM);
            }
        };

        // --- Step 6: mmap the host fd into the SHM region ---

        if let Err(e) = unsafe { self.shm.map_host_fd(region.offset, length, host_map_fd) } {
            log::error!("NV_ESC_RM_MAP_MEMORY: map_host_fd failed: {}", e);
            return self.write_error_resp(resp_buf, Status::IoctlFailed, cookie, libc::ENOMEM);
        }

        log::info!(
            "NV_ESC_RM_MAP_MEMORY: allocated SHM region offset=0x{:x} length=0x{:x} pgprot={:?}",
            region.offset,
            region.length,
            region.pgprot,
        );

        // --- Step 7: Build response with SHM metadata ---

        let hdr = RespHeader {
            status: Status::Ok as u32,
            cookie,
            errno_host: 0,
        };
        let iresp = IoctlResp {
            param_size: param_buf.len() as u32,
            _pad: 0,
            shm_offset: region.offset,
            shm_length: length,
            pgprot: pgprot as u8,
            _pad2: [0; 7],
        };

        let need = size_of::<RespHeader>() + size_of::<IoctlResp>() + param_buf.len();
        if resp_buf.len() < need {
            return self.write_error_resp(resp_buf, Status::BufferTooSmall, cookie, 0);
        }

        let mut off = 0;
        off += write_struct(&mut resp_buf[off..], &hdr);
        off += write_struct(&mut resp_buf[off..], &iresp);
        resp_buf[off..off + param_buf.len()].copy_from_slice(&param_buf);
        off + param_buf.len()
    }

    // ------------------------------------------------------------------
    // Response helpers
    // ------------------------------------------------------------------

    fn write_ioctl_resp(&self, resp_buf: &mut [u8], cookie: u64, param_out: &[u8]) -> usize {
        let hdr = RespHeader {
            status: Status::Ok as u32,
            cookie,
            errno_host: 0,
        };
        let payload = IoctlResp {
            param_size: param_out.len() as u32,
            _pad: 0,
            shm_offset: 0,
            shm_length: 0,
            pgprot: 0,
            _pad2: [0; 7],
        };

        let need = size_of::<RespHeader>() + size_of::<IoctlResp>() + param_out.len();
        if resp_buf.len() < need {
            return self.write_error_resp(resp_buf, Status::BufferTooSmall, cookie, 0);
        }

        let mut off = 0;
        off += write_struct(&mut resp_buf[off..], &hdr);
        off += write_struct(&mut resp_buf[off..], &payload);
        resp_buf[off..off + param_out.len()].copy_from_slice(param_out);
        off + param_out.len()
    }

    fn write_error_resp(
        &self,
        resp_buf: &mut [u8],
        status: Status,
        cookie: u64,
        errno: i32,
    ) -> usize {
        let hdr = RespHeader {
            status: status as u32,
            cookie,
            errno_host: errno,
        };
        if resp_buf.len() < size_of::<RespHeader>() {
            return 0;
        }
        write_struct(resp_buf, &hdr);
        size_of::<RespHeader>()
    }

    // ------------------------------------------------------------------
    // Test helpers
    // ------------------------------------------------------------------

    #[cfg(test)]
    pub fn handle_count(&self) -> usize {
        self.handles.len()
    }
}

// ------------------------------------------------------------------
// Drop: ensure host fds are closed even if teardown() is not called
// ------------------------------------------------------------------

impl Drop for NvidiaBackend {
    fn drop(&mut self) {
        if !self.handles.is_empty() {
            log::warn!(
                "NvidiaBackend dropped with {} handles still open — \
                 call teardown() before dropping for clean shutdown",
                self.handles.len()
            );
            self.handles.drain_all();
        }
    }
}

// ============================================================
// Serialisation helpers
// ============================================================

fn write_ok<P: Copy>(buf: &mut [u8], cookie: u64, payload: &P) -> usize {
    let hdr = RespHeader {
        status: Status::Ok as u32,
        cookie,
        errno_host: 0,
    };
    let sh = size_of::<RespHeader>();
    let sp = size_of::<P>();
    assert!(buf.len() >= sh + sp);
    write_struct(buf, &hdr);
    write_struct(&mut buf[sh..], payload);
    sh + sp
}

fn read_struct<T: Copy>(buf: &[u8], offset: usize) -> T {
    assert!(buf.len() >= offset + size_of::<T>());
    unsafe { (buf.as_ptr().add(offset) as *const T).read_unaligned() }
}

fn write_struct<T: Copy>(buf: &mut [u8], val: &T) -> usize {
    let sz = size_of::<T>();
    assert!(buf.len() >= sz);
    unsafe { (buf.as_mut_ptr() as *mut T).write_unaligned(*val) }
    sz
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn hdr(msg_type: MsgType, cookie: u64) -> Vec<u8> {
        let mut v = vec![0u8; size_of::<MsgHeader>()];
        write_struct(
            &mut v,
            &MsgHeader {
                msg_type: msg_type as u32,
                cookie,
                _pad: 0,
            },
        );
        v
    }

    fn append<T: Copy>(v: &mut Vec<u8>, val: &T) {
        let start = v.len();
        v.resize(start + size_of::<T>(), 0);
        write_struct(&mut v[start..], val);
    }

    fn parse_resp(buf: &[u8]) -> RespHeader {
        read_struct::<RespHeader>(buf, 0)
    }
    fn parse_open_resp(buf: &[u8]) -> OpenResp {
        read_struct::<OpenResp>(buf, size_of::<RespHeader>())
    }

    fn nvidiactl_present() -> bool {
        std::path::Path::new("/dev/nvidiactl").exists()
    }

    // ---- error paths (no GPU required) ----

    #[test]
    fn open_invalid_gpu_index() {
        let mut be = NvidiaBackend::for_test();
        let mut req = hdr(MsgType::Open, 1);
        append(
            &mut req,
            &OpenReq {
                kind: DeviceKind::Gpu as u8,
                index: 200,
                _pad: [0; 6],
            },
        );
        let mut resp = vec![0u8; 64];
        be.dispatch(&req, &mut resp);
        assert_eq!(parse_resp(&resp).status, Status::InvalidDevice as u32);
    }

    #[test]
    fn close_unknown_handle() {
        let mut be = NvidiaBackend::for_test();
        let mut req = hdr(MsgType::Close, 7);
        append(
            &mut req,
            &CloseReq {
                guest_handle: 0xCAFE,
            },
        );
        let mut resp = vec![0u8; 32];
        be.dispatch(&req, &mut resp);
        assert_eq!(parse_resp(&resp).status, Status::BadHandle as u32);
    }

    #[test]
    fn short_request_rejected() {
        let mut be = NvidiaBackend::for_test();
        be.dispatch(&[0u8; 4], &mut vec![0u8; 32]);
        // just must not panic
    }

    // ---- teardown tests (no GPU required) ----

    #[test]
    fn teardown_empties_handles() {
        if !nvidiactl_present() {
            return;
        }
        let mut be = NvidiaBackend::for_test();

        // Open two fds
        for _ in 0..2 {
            let mut req = hdr(MsgType::Open, 1);
            append(
                &mut req,
                &OpenReq {
                    kind: DeviceKind::Ctl as u8,
                    index: 0,
                    _pad: [0; 6],
                },
            );
            let mut resp = vec![0u8; 64];
            be.dispatch(&req, &mut resp);
        }
        assert_eq!(be.handle_count(), 2);

        be.teardown();
        assert_eq!(be.handle_count(), 0);
    }

    #[test]
    fn drop_closes_remaining_handles() {
        if !nvidiactl_present() {
            return;
        }
        // Open a handle, then drop the backend without calling teardown().
        // The Drop impl should drain the table and not panic.
        let mut be = NvidiaBackend::for_test();
        let mut req = hdr(MsgType::Open, 1);
        append(
            &mut req,
            &OpenReq {
                kind: DeviceKind::Ctl as u8,
                index: 0,
                _pad: [0; 6],
            },
        );
        let mut resp = vec![0u8; 64];
        be.dispatch(&req, &mut resp);
        assert_eq!(be.handle_count(), 1);
        drop(be); // must not panic; Drop closes the fd
    }

    // ---- GPU-present round-trip tests ----

    #[test]
    fn open_close_nvidiactl() {
        if !nvidiactl_present() {
            return;
        }
        let mut be = NvidiaBackend::for_test();

        let mut req = hdr(MsgType::Open, 42);
        append(
            &mut req,
            &OpenReq {
                kind: DeviceKind::Ctl as u8,
                index: 0,
                _pad: [0; 6],
            },
        );
        let mut resp = vec![0u8; 64];
        be.dispatch(&req, &mut resp);
        let r = parse_resp(&resp);
        assert_eq!(r.status, Status::Ok as u32);

        let h = parse_open_resp(&resp).guest_handle;
        assert!(h > 0);

        let mut req2 = hdr(MsgType::Close, 43);
        append(&mut req2, &CloseReq { guest_handle: h });
        let mut resp2 = vec![0u8; 32];
        be.dispatch(&req2, &mut resp2);
        assert_eq!(parse_resp(&resp2).status, Status::Ok as u32);
        assert_eq!(be.handle_count(), 0);
    }

    #[test]
    fn check_version_str() {
        if !nvidiactl_present() {
            return;
        }
        let mut be = NvidiaBackend::for_test();

        let mut oreq = hdr(MsgType::Open, 1);
        append(
            &mut oreq,
            &OpenReq {
                kind: DeviceKind::Ctl as u8,
                index: 0,
                _pad: [0; 6],
            },
        );
        let mut oresp = vec![0u8; 64];
        be.dispatch(&oreq, &mut oresp);
        let gh = parse_open_resp(&oresp).guest_handle;
        assert!(gh > 0);

        // nv_ioctl_rm_api_version_t: cmd(4) + reply(4) + versionString(64) = 72 bytes
        let param_size: u32 = 72;
        let mut ireq = hdr(MsgType::Ioctl, 2);
        // NV_ESC_CHECK_VERSION_STR = NV_IOCTL_BASE + 10 = 210
        // _IOWR('F', 210, 72) = (3 << 30) | (72 << 16) | (0x46 << 8) | 210
        append(
            &mut ireq,
            &IoctlReq {
                guest_handle: gh,
                request: abi::ioctl::_IOWR(abi::ioctl::NV_ESC_CHECK_VERSION_STR, param_size),
                param_size,
                _pad: 0,
            },
        );
        // First 4 bytes = cmd field. Set to '2' (0x32) for query mode.
        let mut params = vec![0u8; param_size as usize];
        params[0] = 0x32;
        ireq.extend(params);

        let mut iresp = vec![0u8; 512];
        be.dispatch(&ireq, &mut iresp);
        let r = parse_resp(&iresp);
        assert!(
            r.status == Status::Ok as u32 || r.status == Status::IoctlFailed as u32,
            "unexpected status {}",
            r.status
        );
    }

    /// NV_ESC_RM_MAP_MEMORY round-trip.
    ///
    /// We can't test a real mapping without a valid RM client/device/memory
    /// triple, but we CAN test that:
    ///   1. The dispatch path is reached (not hitting "unhandled escape").
    ///   2. The embedded FD is translated correctly.
    ///   3. The host ioctl failure is reported cleanly (since we don't have
    ///      valid RM handles, the host driver will reject the call).
    #[test]
    fn map_memory_rejects_bad_fd_handle() {
        let mut be = NvidiaBackend::for_test();

        // We need an open nvidiactl fd as the "outer" fd for the ioctl.
        if !nvidiactl_present() {
            return;
        }

        // Open nvidiactl.
        let mut oreq = hdr(MsgType::Open, 1);
        append(
            &mut oreq,
            &OpenReq {
                kind: DeviceKind::Ctl as u8,
                index: 0,
                _pad: [0; 6],
            },
        );
        let mut oresp = vec![0u8; 64];
        be.dispatch(&oreq, &mut oresp);
        let gh = parse_open_resp(&oresp).guest_handle;
        assert!(gh > 0);

        // Build a NV_ESC_RM_MAP_MEMORY ioctl with a bogus embedded FD handle.
        // IoctlNVOS33ParametersWithFD = 56 bytes.
        let param_size: u32 = 56;
        let mut ireq = hdr(MsgType::Ioctl, 2);
        append(
            &mut ireq,
            &IoctlReq {
                guest_handle: gh,
                request: abi::ioctl::_IOWR(abi::ioctl::NV_ESC_RM_MAP_MEMORY, param_size),
                param_size,
                _pad: 0,
            },
        );

        // 56 bytes of zeroed params — the embedded FD at offset 48 is 0,
        // which is not a valid guest handle.
        let mut params = vec![0u8; param_size as usize];
        // Write a bogus FD handle (0xDEAD) at offset 48.
        params[48..52].copy_from_slice(&0xDEADu32.to_le_bytes());
        ireq.extend(params);

        let mut iresp = vec![0u8; 512];
        be.dispatch(&ireq, &mut iresp);
        let r = parse_resp(&iresp);

        // Should fail with BadHandle since 0xDEAD is not in the handle table.
        assert_eq!(r.status, Status::BadHandle as u32);
    }

    #[test]
    fn map_memory_translates_fd_and_forwards() {
        if !nvidiactl_present() {
            return;
        }

        let mut be = NvidiaBackend::for_test();

        // Open nvidiactl — this is both the "outer" fd and the "map" fd.
        let mut oreq = hdr(MsgType::Open, 1);
        append(
            &mut oreq,
            &OpenReq {
                kind: DeviceKind::Ctl as u8,
                index: 0,
                _pad: [0; 6],
            },
        );
        let mut oresp = vec![0u8; 64];
        be.dispatch(&oreq, &mut oresp);
        let ctl_handle = parse_open_resp(&oresp).guest_handle;

        // Open a second nvidiactl fd to use as the embedded map FD.
        let mut oreq2 = hdr(MsgType::Open, 2);
        append(
            &mut oreq2,
            &OpenReq {
                kind: DeviceKind::Ctl as u8,
                index: 0,
                _pad: [0; 6],
            },
        );
        let mut oresp2 = vec![0u8; 64];
        be.dispatch(&oreq2, &mut oresp2);
        let map_handle = parse_open_resp(&oresp2).guest_handle;

        // Build IoctlNVOS33ParametersWithFD with the map_handle as embedded FD.
        let param_size: u32 = 56;
        let mut ireq = hdr(MsgType::Ioctl, 3);
        append(
            &mut ireq,
            &IoctlReq {
                guest_handle: ctl_handle,
                request: abi::ioctl::_IOWR(abi::ioctl::NV_ESC_RM_MAP_MEMORY, param_size),
                param_size,
                _pad: 0,
            },
        );

        let mut params = vec![0u8; param_size as usize];
        // Embedded FD at offset 48 = map_handle.
        params[48..52].copy_from_slice(&(map_handle as u32).to_le_bytes());
        ireq.extend(params);

        let mut iresp = vec![0u8; 512];
        be.dispatch(&ireq, &mut iresp);
        let r = parse_resp(&iresp);

        // The host ioctl will fail (we have no valid RM objects) but the
        // dispatch path should reach the host ioctl — so we expect either
        // IoctlFailed (host rejected it) or Ok (unlikely without valid handles).
        // The key thing: it should NOT be BadHandle, proving FD translation worked.
        assert_ne!(
            r.status,
            Status::BadHandle as u32,
            "FD translation should have succeeded"
        );
    }
}
