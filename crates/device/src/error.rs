// crates/device/src/error.rs

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DeviceError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("descriptor buffer too small: need {need}, have {have}")]
    BufferTooSmall { need: usize, have: usize },

    #[error("unknown guest handle {0}")]
    BadHandle(u64),

    #[error("unknown message type {0}")]
    UnknownMsgType(u32),

    #[error("invalid device kind {0}")]
    InvalidDeviceKind(u8),

    #[error("GPU index {0} out of range")]
    GpuIndexOutOfRange(u8),
}

pub type Result<T> = std::result::Result<T, DeviceError>;
