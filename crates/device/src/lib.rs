// crates/device/src/lib.rs
//
// VMM backend for virtio-gpu-nv.
//
// Integrates with libkrun's virtio device infrastructure.  The backend holds
// real host file descriptors for `/dev/nvidia*` and dispatches messages
// received from the guest driver over virtqueues.

pub mod error;
pub mod handle_table;
pub mod nvidia;
pub mod shm;
pub mod virtio;
