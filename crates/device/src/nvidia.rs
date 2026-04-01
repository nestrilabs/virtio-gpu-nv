// crates/device/src/nvidia.rs
//
// Core backend logic: message dispatch, handle table, host fd management.
//
// `NvidiaBackend` is the main object.  The virtio layer (virtio.rs) calls
// `dispatch()` for each descriptor chain received from the guest.

use std::ffi::CString;
use std::os::fd::{FromRawFd, OwnedFd};

use protocol::messages::*;

use crate::error::{DeviceError, Result};
use crate::handle_table::HandleTable;
use crate::shm::ShmAllocator;

// ---------------------------------------------------------------------------
// Device path helpers
// ---------------------------------------------------------------------------

/// Maximum number of /dev/nvidia# GPU devices we support.
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

// ---------------------------------------------------------------------------
// NvidiaBackend
// ---------------------------------------------------------------------------

pub struct NvidiaBackend {
    handles: HandleTable,
    _shm:    ShmAllocator,   // will be used in Phase 3
}

impl NvidiaBackend {
    /// `shm_bar_size`: size in bytes of the shared memory BAR (must be
    /// page-aligned).  Use 0 to disable the SHM allocator for Phase 1.
    pub fn new(shm_bar_size: u64) -> Self {
        let shm_bar_size = if shm_bar_size == 0 { 4096 } else { shm_bar_size };
        Self {
            handles: HandleTable::new(),
            _shm:    ShmAllocator::new(shm_bar_size),
        }
    }

    // -----------------------------------------------------------------------
    // Top-level dispatch
    // -----------------------------------------------------------------------

    /// Dispatch one message from the guest.
    ///
    /// `req_buf`:  readable part of the descriptor chain (MsgHeader + payload).
    /// `resp_buf`: writable part — caller fills this with the serialized response.
    ///
    /// Returns the number of bytes written into `resp_buf`.
    pub fn dispatch(&mut self, req_buf: &[u8], resp_buf: &mut [u8]) -> usize {
        // --- parse common header ---
        if req_buf.len() < size_of::<MsgHeader>() {
            log::error!("request too short: {} bytes", req_buf.len());
            return self.write_error_resp(resp_buf, Status::InvalidMsgType, 0, 0);
        }
        let hdr = read_struct::<MsgHeader>(req_buf, 0);

        let msg_type = match hdr.msg_type {
            t if t == MsgType::Open  as u32 => MsgType::Open,
            t if t == MsgType::Close as u32 => MsgType::Close,
            t if t == MsgType::Ioctl as u32 => MsgType::Ioctl,
            other => {
                log::warn!("unknown msg_type {}", other);
                return self.write_error_resp(resp_buf, Status::InvalidMsgType, hdr.cookie, 0);
            }
        };

        let payload = &req_buf[size_of::<MsgHeader>()..];

        match msg_type {
            MsgType::Open  => self.handle_open(hdr.cookie, payload, resp_buf),
            MsgType::Close => self.handle_close(hdr.cookie, payload, resp_buf),
            MsgType::Ioctl => self.handle_ioctl(hdr.cookie, payload, resp_buf),
        }
    }

    // -----------------------------------------------------------------------
    // OPEN handler
    // -----------------------------------------------------------------------

    fn handle_open(&mut self, cookie: u64, payload: &[u8], resp_buf: &mut [u8]) -> usize {
        if payload.len() < size_of::<OpenReq>() {
            return self.write_error_resp(resp_buf, Status::InvalidDevice, cookie, 0);
        }
        let req = read_struct::<OpenReq>(payload, 0);

        let path = match device_path(req.kind, req.index) {
            Ok(p)  => p,
            Err(e) => {
                log::warn!("handle_open: bad device: {}", e);
                return self.write_error_resp(resp_buf, Status::InvalidDevice, cookie, 0);
            }
        };

        // Open the real host device.
        let raw_fd = unsafe {
            libc::open(path.as_ptr(), libc::O_RDWR | libc::O_CLOEXEC)
        };

        if raw_fd < 0 {
            let err = std::io::Error::last_os_error();
            log::warn!("open({:?}) failed: {}", path, err);
            let errno = err.raw_os_error().unwrap_or(0);
            return self.write_error_resp(resp_buf, Status::OpenFailed, cookie, errno);
        }

        let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
        let guest_handle = self.handles.insert(fd);

        log::debug!("opened {:?} → guest_handle={}", path, guest_handle);

        // Write success response.
        let resp_hdr = RespHeader {
            status: Status::Ok as u32,
            cookie,
            errno_host: 0,
        };
        let resp_payload = OpenResp { guest_handle };

        write_structs(resp_buf, &resp_hdr, &resp_payload)
    }

    // -----------------------------------------------------------------------
    // CLOSE handler
    // -----------------------------------------------------------------------

    fn handle_close(&mut self, cookie: u64, payload: &[u8], resp_buf: &mut [u8]) -> usize {
        if payload.len() < size_of::<CloseReq>() {
            return self.write_error_resp(resp_buf, Status::BadHandle, cookie, 0);
        }
        let req = read_struct::<CloseReq>(payload, 0);

        match self.handles.remove(req.guest_handle) {
            Ok(()) => {
                log::debug!("closed guest_handle={}", req.guest_handle);
                let resp_hdr = RespHeader {
                    status: Status::Ok as u32,
                    cookie,
                    errno_host: 0,
                };
                write_structs(resp_buf, &resp_hdr, &CloseResp { _pad: 0 })
            }
            Err(_) => {
                self.write_error_resp(resp_buf, Status::BadHandle, cookie, 0)
            }
        }
    }

    // -----------------------------------------------------------------------
    // IOCTL handler (stub — Phase 2)
    // -----------------------------------------------------------------------

    fn handle_ioctl(&mut self, cookie: u64, _payload: &[u8], resp_buf: &mut [u8]) -> usize {
        // Phase 2 will implement full ioctl dispatch.
        log::debug!("ioctl stub — not yet implemented");
        self.write_error_resp(resp_buf, Status::InvalidMsgType, cookie, libc::ENOSYS)
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

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
        let min = size_of::<RespHeader>();
        if resp_buf.len() < min {
            return 0;
        }
        write_struct(resp_buf, &hdr);
        min
    }

    /// Return the number of open guest handles (for tests).
    #[cfg(test)]
    pub fn handle_count(&self) -> usize {
        self.handles.len()
    }
}

// ---------------------------------------------------------------------------
// Low-level serialisation helpers
// ---------------------------------------------------------------------------

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

fn write_structs<A: Copy, B: Copy>(buf: &mut [u8], a: &A, b: &B) -> usize {
    let sa = size_of::<A>();
    let sb = size_of::<B>();
    assert!(buf.len() >= sa + sb);
    write_struct(buf, a);
    write_struct(&mut buf[sa..], b);
    sa + sb
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_backend() -> NvidiaBackend {
        NvidiaBackend::new(4096)
    }

    fn make_open_req(kind: DeviceKind, index: u8) -> Vec<u8> {
        let mut buf = vec![0u8; size_of::<MsgHeader>() + size_of::<OpenReq>()];
        write_struct(
            &mut buf,
            &MsgHeader { msg_type: MsgType::Open as u32, cookie: 42, _pad: 0 },
        );
        write_struct(
            &mut buf[size_of::<MsgHeader>()..],
            &OpenReq { kind: kind as u8, index, _pad: [0; 6] },
        );
        buf
    }

    fn make_close_req(handle: u64) -> Vec<u8> {
        let mut buf = vec![0u8; size_of::<MsgHeader>() + size_of::<CloseReq>()];
        write_struct(
            &mut buf,
            &MsgHeader { msg_type: MsgType::Close as u32, cookie: 99, _pad: 0 },
        );
        write_struct(&mut buf[size_of::<MsgHeader>()..], &CloseReq { guest_handle: handle });
        buf
    }

    fn parse_resp_hdr(buf: &[u8]) -> RespHeader {
        read_struct::<RespHeader>(buf, 0)
    }

    fn parse_open_resp(buf: &[u8]) -> OpenResp {
        read_struct::<OpenResp>(buf, size_of::<RespHeader>())
    }

    // ------------------------------------------------------------------
    // Open /dev/nvidiactl
    // ------------------------------------------------------------------

    /// Happy-path: open nvidiactl on a host that actually has NVIDIA drivers.
    /// Skipped automatically if /dev/nvidiactl is absent (CI without GPU).
    #[test]
    fn open_nvidiactl_if_present() {
        if !std::path::Path::new("/dev/nvidiactl").exists() {
            eprintln!("SKIP: /dev/nvidiactl not found on this host");
            return;
        }

        let mut be = make_backend();
        let req = make_open_req(DeviceKind::Ctl, 0);
        let mut resp = vec![0u8; 64];
        let n = be.dispatch(&req, &mut resp);
        assert!(n >= size_of::<RespHeader>() + size_of::<OpenResp>());

        let hdr = parse_resp_hdr(&resp);
        assert_eq!(hdr.status, Status::Ok as u32);
        assert_eq!(hdr.cookie, 42);

        let open_resp = parse_open_resp(&resp);
        assert!(open_resp.guest_handle > 0);
        assert_eq!(be.handle_count(), 1);

        // Close it.
        let req2 = make_close_req(open_resp.guest_handle);
        let mut resp2 = vec![0u8; 32];
        let n2 = be.dispatch(&req2, &mut resp2);
        assert!(n2 >= size_of::<RespHeader>());
        let hdr2 = parse_resp_hdr(&resp2);
        assert_eq!(hdr2.status, Status::Ok as u32);
        assert_eq!(be.handle_count(), 0);
    }

    // ------------------------------------------------------------------
    // Error paths (no GPU required)
    // ------------------------------------------------------------------

    #[test]
    fn open_invalid_device_kind() {
        let mut be = make_backend();
        let req = make_open_req(DeviceKind::Gpu, 255); // index out of range
        let mut resp = vec![0u8; 64];
        be.dispatch(&req, &mut resp);
        let hdr = parse_resp_hdr(&resp);
        assert_eq!(hdr.status, Status::InvalidDevice as u32);
    }

    #[test]
    fn close_bad_handle() {
        let mut be = make_backend();
        let req = make_close_req(0xDEADBEEF);
        let mut resp = vec![0u8; 32];
        be.dispatch(&req, &mut resp);
        let hdr = parse_resp_hdr(&resp);
        assert_eq!(hdr.status, Status::BadHandle as u32);
    }

    #[test]
    fn unknown_msg_type() {
        let mut be = make_backend();
        let mut req = vec![0u8; size_of::<MsgHeader>()];
        write_struct(&mut req, &MsgHeader { msg_type: 0xFF, cookie: 1, _pad: 0 });
        let mut resp = vec![0u8; 32];
        be.dispatch(&req, &mut resp);
        let hdr = parse_resp_hdr(&resp);
        assert_eq!(hdr.status, Status::InvalidMsgType as u32);
    }

    #[test]
    fn request_too_short() {
        let mut be = make_backend();
        let req = vec![0u8; 4]; // shorter than MsgHeader
        let mut resp = vec![0u8; 32];
        be.dispatch(&req, &mut resp);
        let hdr = parse_resp_hdr(&resp);
        assert_eq!(hdr.status, Status::InvalidMsgType as u32);
    }
}
