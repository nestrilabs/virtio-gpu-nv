// crates/device/src/shm.rs
//
// SHM BAR region allocator.
//
// The SHM BAR is a contiguous region in the VMM address space that is exposed
// to the guest via a KVM memslot.  When the backend handles a mapping ioctl
// (e.g. NV_ESC_RM_MAP_MEMORY) it:
//   1. Calls the host ioctl to get the mapping.
//   2. Allocates a region in the SHM BAR.
//   3. mmap(MAP_SHARED|MAP_FIXED)s the host fd into that region.
//   4. Returns the SHM offset + length to the guest driver.
//
// The guest driver then calls remap_pfn_range() to expose the SHM range to
// userspace.
//
// Phase 1: the allocator exists but is not exercised (no mapping ioctls yet).

use crate::error::{DeviceError, Result};

/// A single allocated region within the SHM BAR.
#[derive(Debug)]
pub struct ShmRegion {
    /// Byte offset from the start of the SHM BAR.
    pub offset: u64,
    /// Byte length of the region.
    pub length: u64,
}

/// Bump allocator for the SHM BAR.
///
/// A proper allocator would reclaim regions when a mapping is destroyed
/// (Phase 3+).  For now, a simple bump pointer is sufficient to get
/// mapping ioctls working.
pub struct ShmAllocator {
    /// Total size of the BAR in bytes.
    bar_size: u64,
    /// Next free byte offset.
    cursor: u64,
}

impl ShmAllocator {
    /// `bar_size` must be page-aligned and large enough to hold all concurrent
    /// GPU mappings.  A typical value is 256 MiB or 1 GiB.
    pub fn new(bar_size: u64) -> Self {
        assert!(bar_size % 4096 == 0, "bar_size must be page-aligned");
        Self {
            bar_size,
            cursor: 0,
        }
    }

    /// Allocate a region of `length` bytes.
    ///
    /// `length` is rounded up to the next page boundary.
    /// Returns `Err` if the BAR is full.
    pub fn alloc(&mut self, length: u64) -> Result<ShmRegion> {
        let aligned = align_up(length, 4096);
        if self.cursor + aligned > self.bar_size {
            return Err(DeviceError::Io(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "SHM BAR full",
            )));
        }
        let offset = self.cursor;
        self.cursor += aligned;
        Ok(ShmRegion { offset, length })
    }

    /// Bytes remaining in the BAR.
    pub fn free_bytes(&self) -> u64 {
        self.bar_size - self.cursor
    }
}

fn align_up(v: u64, align: u64) -> u64 {
    (v + align - 1) & !(align - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_alloc() {
        let mut a = ShmAllocator::new(4096 * 4);
        let r1 = a.alloc(100).unwrap();
        assert_eq!(r1.offset, 0);
        assert_eq!(r1.length, 100);
        let r2 = a.alloc(4096).unwrap();
        assert_eq!(r2.offset, 4096); // first alloc rounded up to one page
    }

    #[test]
    fn full_bar() {
        let mut a = ShmAllocator::new(4096);
        a.alloc(4096).unwrap();
        assert!(a.alloc(1).is_err());
    }
}
