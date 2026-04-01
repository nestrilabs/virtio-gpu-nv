// crates/abi/src/lib.rs
//
// NVIDIA kernel driver ABI definitions.
//
// Ported from gVisor's pkg/abi/nvgpu/ (Apache-2.0).
// Phase 1 carries only the minimum needed; Phase 2+ will expand this.

pub mod types;
pub mod ioctl;
pub mod version;
pub mod versions;
