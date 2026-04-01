// crates/device/src/mmap.rs
//
// Tracks active GPU→SHM mappings so the backend can unmap them when the
// guest calls NV_ESC_RM_UNMAP_MEMORY or closes the file.
//
// Phase 2: data structure only, no entries are created yet.
// Phase 3: mapping ioctls populate this table.

use std::collections::HashMap;

/// One active mmap region in the SHM BAR.
#[derive(Debug)]
pub struct MmapEntry {
    /// Byte offset from the start of the SHM BAR (what the guest sees).
    pub shm_offset: u64,
    /// Length of the mapped region in bytes.
    pub length: u64,
    /// Host virtual address where the mapping lives (for munmap on teardown).
    pub host_va: usize,
    /// Page-protection kind: 0=WB, 1=WC, 2=UC.
    pub pgprot: u8,
}

/// Indexed by the guest handle of the file on which the mapping was created.
pub struct MmapContext {
    entries: HashMap<u64 /* shm_offset */, MmapEntry>,
}

impl MmapContext {
    pub fn new() -> Self {
        Self { entries: HashMap::new() }
    }

    pub fn insert(&mut self, entry: MmapEntry) {
        self.entries.insert(entry.shm_offset, entry);
    }

    pub fn remove(&mut self, shm_offset: u64) -> Option<MmapEntry> {
        self.entries.remove(&shm_offset)
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for MmapContext {
    fn default() -> Self { Self::new() }
}
