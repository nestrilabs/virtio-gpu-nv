// crates/device/src/virtio.rs
//
// Virtio device trait and queue-processing loop.
//
// This module provides the glue between libkrun's virtio infrastructure and
// `NvidiaBackend`.  In Phase 1 we define the trait and a simple polling loop;
// actual libkrun integration is wired up later.
//
// The virtio spec (section 2.6) defines the split virtqueue layout:
//   - Descriptor table: array of descriptors (addr, len, flags, next)
//   - Available ring: guest→host notification ring
//   - Used ring:      host→guest notification ring
//
// For our single virtqueue:
//   - Each descriptor chain represents one operation.
//   - Readable descriptors = request buffer.
//   - Writable descriptors = response buffer.

use crate::nvidia::NvidiaBackend;

// ---------------------------------------------------------------------------
// Virtio descriptor (as seen by the backend after GPA→HVA translation)
// ---------------------------------------------------------------------------

/// A single scatter-gather entry after the VMM has resolved guest physical
/// addresses to host virtual addresses.
pub struct Descriptor<'a> {
    pub data:     &'a [u8],
    pub writable: bool,
}

/// A resolved descriptor chain for one operation.
pub struct DescChain<'a> {
    pub descriptors: Vec<Descriptor<'a>>,
}

impl<'a> DescChain<'a> {
    /// Collect all readable bytes in order into a single contiguous slice.
    ///
    /// We allocate a Vec here because the readable part may be split across
    /// multiple physical pages; in practice NVIDIA ioctl params fit in a
    /// single page so this is rarely needed.
    pub fn readable_bytes(&self) -> Vec<u8> {
        self.descriptors
            .iter()
            .filter(|d| !d.writable)
            .flat_map(|d| d.data.iter().copied())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// VirtioDevice trait
// ---------------------------------------------------------------------------

/// Implemented by the libkrun integration layer to give `NvidiaDevice` access
/// to the virtqueue machinery.
pub trait VirtioDevice {
    /// Block until a new descriptor chain is available, then call `f` with:
    ///   - the readable buffer (request),
    ///   - the writable buffer (response),
    ///
    /// `f` should write into `resp` and return the number of bytes written.
    /// The implementation posts the used element back to the guest.
    fn process_queue<F>(&mut self, f: F)
    where
        F: FnMut(&[u8], &mut [u8]) -> usize;
}

// ---------------------------------------------------------------------------
// NvidiaDevice: the public API for libkrun integration
// ---------------------------------------------------------------------------

/// Wraps `NvidiaBackend` and drives the virtqueue processing loop.
pub struct NvidiaDevice {
    backend: NvidiaBackend,
}

impl NvidiaDevice {
    pub fn new(shm_bar_size: u64) -> Self {
        Self {
            backend: NvidiaBackend::new(shm_bar_size),
        }
    }

    /// Run the device loop using the provided virtio transport.
    ///
    /// Blocks indefinitely; call from a dedicated thread.
    pub fn run<V: VirtioDevice>(&mut self, mut transport: V) {
        loop {
            let backend = &mut self.backend;
            transport.process_queue(|req, resp| backend.dispatch(req, resp));
        }
    }
}
