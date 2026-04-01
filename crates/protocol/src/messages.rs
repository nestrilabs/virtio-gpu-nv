// crates/protocol/src/messages.rs
//
// Wire message types for virtio-gpu-nv.
//
// Descriptor chain layout (one chain per operation):
//
//   [readable: MsgHeader + payload] → [writable: RespHeader + payload]
//
// The guest driver:
//   1. Fills in the readable buffer (header + request payload).
//   2. Adds the writable buffer to the chain.
//   3. Posts the chain to the virtqueue and waits for a used-ring notification.
//   4. Reads the writable buffer (header + response payload).
//
// The backend:
//   1. Receives the readable buffer.
//   2. Dispatches on `MsgHeader::msg_type`.
//   3. Writes the response into the writable buffer.
//   4. Pushes the chain back to the used ring.

// ---------------------------------------------------------------------------
// Message type discriminants
// ---------------------------------------------------------------------------

/// Identifies the kind of message in a `MsgHeader`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MsgType {
    /// Guest → host: open a `/dev/nvidia*` device.
    Open  = 1,
    /// Guest → host: close a previously opened handle.
    Close = 2,
    /// Guest → host: forward a raw ioctl.
    Ioctl = 3,
}

// ---------------------------------------------------------------------------
// Status codes (response header)
// ---------------------------------------------------------------------------

/// Status codes returned by the backend in `RespHeader::status`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Ok    = 0,
    /// The `msg_type` field was not recognized.
    InvalidMsgType = 1,
    /// The requested device path was invalid or not allowed.
    InvalidDevice  = 2,
    /// The host `open(2)` call failed; see `errno_host` for the host errno.
    OpenFailed     = 3,
    /// The guest handle was not found in the handle table.
    BadHandle      = 4,
    /// The host `ioctl(2)` call failed; see `errno_host` for the host errno.
    IoctlFailed    = 5,
    /// A buffer was too small to hold the response payload.
    BufferTooSmall = 6,
}

// ---------------------------------------------------------------------------
// Common header (request side)
// ---------------------------------------------------------------------------

/// Every request starts with this header.
///
/// Total readable buffer = `sizeof(MsgHeader)` + message-specific payload.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MsgHeader {
    /// Discriminant — one of the `MsgType` values.
    pub msg_type: u32,
    /// Cookie chosen by the guest driver; echoed back in `RespHeader::cookie`.
    /// Used to match responses to pending requests when multiple virtqueues
    /// are in flight (not needed for a single-queue design, but good practice).
    pub cookie: u64,
    /// Padding to make the header a multiple of 8 bytes.
    pub _pad: u32,
}

// ---------------------------------------------------------------------------
// Common header (response side)
// ---------------------------------------------------------------------------

/// Every response starts with this header.
///
/// Total writable buffer = `sizeof(RespHeader)` + message-specific payload.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RespHeader {
    /// One of the `Status` values.
    pub status: u32,
    /// Echoed from `MsgHeader::cookie`.
    pub cookie: u64,
    /// Host `errno` value when `status` indicates a syscall failure,
    /// zero otherwise.
    pub errno_host: i32,
}

// ---------------------------------------------------------------------------
// OPEN
// ---------------------------------------------------------------------------

/// Which `/dev/nvidia*` node to open.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    /// `/dev/nvidiactl`
    Ctl = 0,
    /// `/dev/nvidia0` … `/dev/nvidia7`
    Gpu = 1,
    /// `/dev/nvidia-uvm`
    Uvm = 2,
}

/// Request payload for `MsgType::Open`.
///
/// Readable buffer layout:
///   `MsgHeader` | `OpenReq`
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OpenReq {
    /// Which device family to open.
    pub kind: u8,
    /// GPU index (0-based) when `kind == DeviceKind::Gpu`, ignored otherwise.
    pub index: u8,
    /// Reserved; must be zero.
    pub _pad: [u8; 6],
}

/// Response payload for `MsgType::Open`.
///
/// Writable buffer layout:
///   `RespHeader` | `OpenResp`
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OpenResp {
    /// Opaque handle the guest driver should use for subsequent requests.
    /// Valid only when `RespHeader::status == Status::Ok`.
    pub guest_handle: u64,
}

// ---------------------------------------------------------------------------
// CLOSE
// ---------------------------------------------------------------------------

/// Request payload for `MsgType::Close`.
///
/// Readable buffer layout:
///   `MsgHeader` | `CloseReq`
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CloseReq {
    /// Handle returned by a previous `OpenResp`.
    pub guest_handle: u64,
}

/// Response payload for `MsgType::Close`.
///
/// Writable buffer layout:
///   `RespHeader` | `CloseResp`
///
/// Currently empty; `RespHeader::status` carries all the information needed.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CloseResp {
    pub _pad: u64,
}

// ---------------------------------------------------------------------------
// IOCTL (Phase 2 — defined here so the header can be generated now)
// ---------------------------------------------------------------------------

/// Request payload for `MsgType::Ioctl`.
///
/// Readable buffer layout:
///   `MsgHeader` | `IoctlReq` | raw ioctl parameter bytes
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct IoctlReq {
    /// Handle of the open file on which to issue the ioctl.
    pub guest_handle: u64,
    /// The ioctl request number (e.g. `NV_ESC_CHECK_VERSION_STR`).
    pub request: u64,
    /// Number of raw parameter bytes that follow this struct.
    pub param_size: u32,
    pub _pad: u32,
}

/// Response payload for `MsgType::Ioctl`.
///
/// Writable buffer layout:
///   `RespHeader` | `IoctlResp` | raw ioctl parameter bytes (updated by host)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct IoctlResp {
    /// Number of raw parameter bytes that follow this struct.
    pub param_size: u32,
    pub _pad: u32,
    /// When the host ioctl produced a new mmap region (e.g. NV_ESC_RM_MAP_MEMORY),
    /// the backend fills in the SHM BAR offset and byte length here.
    /// Both are zero for non-mapping ioctls.
    pub shm_offset: u64,
    pub shm_length: u64,
    /// Page-protection flags for `remap_pfn_range()` in the guest driver.
    /// Encoding: 0 = WB (write-back), 1 = WC (write-combining), 2 = UC (uncached).
    pub pgprot: u8,
    pub _pad2: [u8; 7],
}

// ---------------------------------------------------------------------------
// Size assertions (compile-time, no_std compatible)
// ---------------------------------------------------------------------------
//
// These catch accidental padding changes that would break the C header.

const _: () = {
    assert!(core::mem::size_of::<MsgHeader>()  == 16);
    assert!(core::mem::size_of::<RespHeader>() == 16);
    assert!(core::mem::size_of::<OpenReq>()    ==  8);
    assert!(core::mem::size_of::<OpenResp>()   ==  8);
    assert!(core::mem::size_of::<CloseReq>()   ==  8);
    assert!(core::mem::size_of::<CloseResp>()  ==  8);
    assert!(core::mem::size_of::<IoctlReq>()   == 24);
    assert!(core::mem::size_of::<IoctlResp>()  == 32);
};
