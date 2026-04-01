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

        use abi::ioctl::*;
        match escape {
            NV_ESC_CHECK_VERSION_STR
            | NV_ESC_CARD_INFO
            | NV_ESC_STATUS_CODE
            | NV_ESC_RM_FREE
            | NV_ESC_RM_UNMAP_MEMORY
            | NV_ESC_RM_DUP_OBJECT
            | NV_ESC_RM_SHARE
            | NV_ESC_RM_UPDATE_DEVICE_MAPPING_INFO => {
                self.dispatch_simple(cookie, host_fd, ireq.request, param_in, resp_buf)
            }

            NV_ESC_REGISTER_FD | NV_ESC_ALLOC_OS_EVENT | NV_ESC_FREE_OS_EVENT => {
                self.dispatch_fd_carrying(cookie, host_fd, ireq.request, escape, param_in, resp_buf)
            }

            NV_ESC_RM_MAP_MEMORY => {
                // Phase 3: allocate from shm, mmap host fd, return offset.
                // pgprot must be determined per-mapping from the RM mapping
                // flags in NVOS03_PARAMETERS::flags — see nvproxy
                // frontend.go:rmMapMemory() for the UC/WC/WB classification.
                log::debug!("NV_ESC_RM_MAP_MEMORY: stub (Phase 3)");
                self.write_error_resp(resp_buf, Status::IoctlFailed, cookie, libc::ENOSYS)
            }

            NV_ESC_RM_CONTROL | NV_ESC_RM_ALLOC | NV_ESC_RM_ALLOC_MEMORY => {
                log::debug!("escape 0x{:02x}: nested dispatch stub (Phase 3)", escape);
                self.write_error_resp(resp_buf, Status::IoctlFailed, cookie, libc::ENOSYS)
            }

            other => {
                log::warn!("unhandled ioctl escape 0x{:02x}", other);
                self.write_error_resp(resp_buf, Status::IoctlFailed, cookie, libc::ENOTTY)
            }
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
            NV_ESC_REGISTER_FD => 0,
            NV_ESC_ALLOC_OS_EVENT => 16,
            NV_ESC_FREE_OS_EVENT => 0,
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

        let param_size: u32 = 68;
        let mut ireq = hdr(MsgType::Ioctl, 2);
        append(
            &mut ireq,
            &IoctlReq {
                guest_handle: gh,
                request: ((3u64 << 30) | (68 << 16) | (0x46 << 8) | 0x25),
                param_size,
                _pad: 0,
            },
        );
        ireq.extend(vec![0u8; param_size as usize]);

        let mut iresp = vec![0u8; 512];
        be.dispatch(&ireq, &mut iresp);
        let r = parse_resp(&iresp);
        assert!(
            r.status == Status::Ok as u32 || r.status == Status::IoctlFailed as u32,
            "unexpected status {}",
            r.status
        );
    }
}
