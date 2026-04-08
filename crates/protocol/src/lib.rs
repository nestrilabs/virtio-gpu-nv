// crates/protocol/src/lib.rs
//
// Wire protocol shared between the VMM backend (Rust) and the guest kernel
// driver (C).  Every type here has a matching definition in
// `guest-driver/virtio_gpu_nv.h`.
//
// Layout rules
// ============
// * All structs are `#[repr(C)]` so the compiler never reorders fields.
// * All integers are little-endian on the wire; the host and guest are both
//   x86-64 so no byte-swapping is needed, but the field types signal intent.
// * Every request starts with a `MsgHeader`; every response starts with a
//   `RespHeader`.  The guest driver allocates one descriptor chain per
//   operation: readable part = request, writable part = response.

#![no_std]

pub mod messages;

pub use messages::*;
