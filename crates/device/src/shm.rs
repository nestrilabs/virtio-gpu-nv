// crates/device/src/shm.rs
//
// SHM BAR region allocator — with per-cache-type zone partitioning
// and memfd backing.
//
// --- Why memfd ---
//
// The SHM BAR must be backed by a file descriptor that can be:
//   1. mmap'd into the VMM's address space (so we can MAP_FIXED host
//      GPU mappings into it).
//   2. Passed to KVM as a memory backend (so the guest can access it
//      via EPT/NPT).
//
// memfd_create() gives us an anonymous file that satisfies both.  We
// ftruncate it to the total BAR size at construction, then mmap the
// whole thing MAP_SHARED.  When the backend handles NV_ESC_RM_MAP_MEMORY,
// it mmap's the host nvidia fd MAP_FIXED into the appropriate offset
// within this region.
//
// --- Zone partitioning ---
//
// See previous comments about UC/WC/WB zones.  The VMM must map each
// zone's physical range in the EPT with the matching memory type.

use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::ptr;

use crate::error::{DeviceError, Result};

/// CPU cache attribute for a SHM region.
///
/// Encoded in `ioctl_resp::pgprot` and in `ShmRegion::pgprot` so the guest
/// driver can call remap_pfn_range() with the correct pgprot_t.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PgprotKind {
    /// Write-back: normal cacheable memory.  Used for system-memory DMA buffers.
    WriteBack = 0,
    /// Write-combining: coalesced uncached writes.  Used for VRAM/framebuffer BARs.
    WriteCombine = 1,
    /// Uncached: no caching, strongly ordered.  Used for MMIO registers/doorbells.
    Uncached = 2,
}

impl PgprotKind {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::WriteBack),
            1 => Some(Self::WriteCombine),
            2 => Some(Self::Uncached),
            _ => None,
        }
    }
}

/// A single allocated region within the SHM BAR.
#[derive(Debug)]
pub struct ShmRegion {
    /// Byte offset from the start of the SHM BAR (what the guest uses as
    /// the mmap offset).
    pub offset: u64,
    /// Byte length of the region (before page-alignment).
    pub length: u64,
    /// Required CPU cache attribute for this region.
    pub pgprot: PgprotKind,
}

/// Zone boundaries within the BAR.
struct Zone {
    /// Absolute byte offset of the start of this zone.
    base: u64,
    /// Total size of this zone in bytes.
    size: u64,
    /// Next free offset within this zone (relative to `base`).
    cursor: u64,
}

impl Zone {
    fn new(base: u64, size: u64) -> Self {
        Self {
            base,
            size,
            cursor: 0,
        }
    }

    fn alloc(&mut self, length: u64) -> Option<u64> {
        let aligned = align_up(length, 4096);
        if self.cursor + aligned > self.size {
            return None;
        }
        let offset = self.base + self.cursor;
        self.cursor += aligned;
        Some(offset)
    }

    fn free_bytes(&self) -> u64 {
        self.size - self.cursor
    }
}

/// BAR zone configuration.
pub struct ZoneConfig {
    /// Size of the UC zone in bytes (must be page-aligned).
    pub uc_size: u64,
    /// Size of the WC zone in bytes (must be page-aligned).
    pub wc_size: u64,
    /// Size of the WB zone in bytes (must be page-aligned).
    /// Total BAR size = uc_size + wc_size + wb_size.
    pub wb_size: u64,
}

impl ZoneConfig {
    /// Default split: 4 MiB UC, 128 MiB WC, 124 MiB WB  (= 256 MiB total).
    pub fn default_256mib() -> Self {
        Self {
            uc_size: 4 * 1024 * 1024,
            wc_size: 128 * 1024 * 1024,
            wb_size: 124 * 1024 * 1024,
        }
    }

    pub fn total(&self) -> u64 {
        self.uc_size + self.wc_size + self.wb_size
    }
}

/// Partitioned bump allocator for the SHM BAR, backed by a memfd.
pub struct ShmAllocator {
    uc: Zone,
    wc: Zone,
    wb: Zone,

    /// The memfd backing the entire BAR.
    memfd: OwnedFd,

    /// Base pointer of the mmap'd region in our address space.
    /// The region spans [base_ptr, base_ptr + total_size).
    base_ptr: *mut u8,

    /// Total size of the BAR (UC + WC + WB).
    total_size: u64,
}

// SAFETY: The memfd and mmap region are owned solely by this struct.
// Access is serialised by the caller (NvidiaBackend holds &mut self).
unsafe impl Send for ShmAllocator {}
unsafe impl Sync for ShmAllocator {}

impl ShmAllocator {
    pub fn new(cfg: ZoneConfig) -> Self {
        assert_eq!(cfg.uc_size % 4096, 0);
        assert_eq!(cfg.wc_size % 4096, 0);
        assert_eq!(cfg.wb_size % 4096, 0);

        let total = cfg.total();
        assert!(total > 0);

        // Create the memfd.
        let name = CString::new("virtio-gpu-nv-shm").unwrap();
        let raw_fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
        assert!(
            raw_fd >= 0,
            "memfd_create failed: {}",
            std::io::Error::last_os_error()
        );
        let memfd = unsafe { OwnedFd::from_raw_fd(raw_fd) };

        // Size the memfd.
        let ret = unsafe { libc::ftruncate(memfd.as_raw_fd(), total as libc::off_t) };
        assert_eq!(
            ret,
            0,
            "ftruncate failed: {}",
            std::io::Error::last_os_error()
        );

        // mmap the entire region MAP_SHARED so MAP_FIXED sub-mappings work.
        let base_ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                total as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                memfd.as_raw_fd(),
                0,
            )
        };
        assert_ne!(
            base_ptr,
            libc::MAP_FAILED,
            "mmap SHM BAR failed: {}",
            std::io::Error::last_os_error()
        );

        let uc_base = 0;
        let wc_base = cfg.uc_size;
        let wb_base = cfg.uc_size + cfg.wc_size;

        Self {
            uc: Zone::new(uc_base, cfg.uc_size),
            wc: Zone::new(wc_base, cfg.wc_size),
            wb: Zone::new(wb_base, cfg.wb_size),
            memfd,
            base_ptr: base_ptr as *mut u8,
            total_size: total,
        }
    }

    pub fn with_default_zones() -> Self {
        Self::new(ZoneConfig::default_256mib())
    }

    /// Allocate `length` bytes from the zone matching `pgprot`.
    ///
    /// Returns `Err` if the zone is full.  Lengths are rounded up to the
    /// next page boundary within the zone.
    pub fn alloc(&mut self, length: u64, pgprot: PgprotKind) -> Result<ShmRegion> {
        let zone = match pgprot {
            PgprotKind::Uncached => &mut self.uc,
            PgprotKind::WriteCombine => &mut self.wc,
            PgprotKind::WriteBack => &mut self.wb,
        };

        match zone.alloc(length) {
            Some(offset) => Ok(ShmRegion {
                offset,
                length,
                pgprot,
            }),
            None => Err(DeviceError::Io(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                format!(
                    "SHM {:?} zone full ({} bytes free, {} requested)",
                    pgprot,
                    zone.free_bytes(),
                    length
                ),
            ))),
        }
    }

    /// mmap a host fd into the SHM BAR at the given offset.
    ///
    /// This is called after `alloc()` to place the actual host GPU mapping
    /// into the SHM region.  Uses MAP_FIXED to overwrite the memfd-backed
    /// page(s) at `offset` with the host fd's mapping.
    ///
    /// # Safety
    ///
    /// `host_fd` must be a valid file descriptor that supports mmap
    /// (e.g. an NVIDIA device fd after NV_ESC_RM_MAP_MEMORY).
    /// `offset` and `length` must be within the BAR and page-aligned.
    pub unsafe fn map_host_fd(&self, offset: u64, length: u64, host_fd: RawFd) -> Result<()> {
        let target = unsafe { self.base_ptr.add(offset as usize) as *mut libc::c_void };
        let ptr = unsafe {
            libc::mmap(
                target,
                length as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_FIXED,
                host_fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            let err = std::io::Error::last_os_error();
            log::error!(
                "SHM map_host_fd: mmap(offset=0x{:x}, len=0x{:x}, fd={}) failed: {}",
                offset,
                length,
                host_fd,
                err
            );
            return Err(DeviceError::Io(err));
        }
        log::debug!(
            "SHM map_host_fd: mapped fd={} at offset=0x{:x} length=0x{:x}",
            host_fd,
            offset,
            length
        );
        Ok(())
    }

    /// Raw fd of the memfd, for passing to KVM as a memory backend.
    pub fn memfd_raw(&self) -> RawFd {
        self.memfd.as_raw_fd()
    }

    /// Base pointer of the mmap'd SHM BAR region.
    pub fn base_ptr(&self) -> *mut u8 {
        self.base_ptr
    }

    pub fn uc_zone_offset(&self) -> u64 {
        self.uc.base
    }
    pub fn wc_zone_offset(&self) -> u64 {
        self.wc.base
    }
    pub fn wb_zone_offset(&self) -> u64 {
        self.wb.base
    }
    pub fn total_size(&self) -> u64 {
        self.total_size
    }
}

impl Drop for ShmAllocator {
    fn drop(&mut self) {
        if !self.base_ptr.is_null() {
            unsafe {
                libc::munmap(self.base_ptr as *mut libc::c_void, self.total_size as usize);
            }
            self.base_ptr = ptr::null_mut();
        }
        // OwnedFd drops the memfd automatically.
    }
}

fn align_up(v: u64, align: u64) -> u64 {
    (v + align - 1) & !(align - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> ZoneConfig {
        ZoneConfig {
            uc_size: 4096 * 2,
            wc_size: 4096 * 4,
            wb_size: 4096 * 2,
        }
    }

    fn small_alloc() -> ShmAllocator {
        ShmAllocator::new(small_cfg())
    }

    #[test]
    fn memfd_is_valid() {
        let a = small_alloc();
        assert!(a.memfd_raw() >= 0);
        assert!(!a.base_ptr().is_null());
        assert_eq!(a.total_size(), 4096 * 8);
    }

    #[test]
    fn zones_dont_overlap() {
        let a = small_alloc();
        // UC ends where WC begins
        assert_eq!(a.uc.base + a.uc.size, a.wc.base);
        // WC ends where WB begins
        assert_eq!(a.wc.base + a.wc.size, a.wb.base);
    }

    #[test]
    fn alloc_correct_zone() {
        let mut a = small_alloc();
        let uc = a.alloc(100, PgprotKind::Uncached).unwrap();
        let wc = a.alloc(100, PgprotKind::WriteCombine).unwrap();
        let wb = a.alloc(100, PgprotKind::WriteBack).unwrap();

        assert_eq!(uc.offset, a.uc.base);
        assert_eq!(wc.offset, a.wc.base);
        assert_eq!(wb.offset, a.wb.base);

        assert_eq!(uc.pgprot, PgprotKind::Uncached);
        assert_eq!(wc.pgprot, PgprotKind::WriteCombine);
        assert_eq!(wb.pgprot, PgprotKind::WriteBack);
    }

    #[test]
    fn alloc_respects_page_alignment() {
        let mut a = small_alloc();
        let r1 = a.alloc(1, PgprotKind::WriteBack).unwrap();
        let r2 = a.alloc(1, PgprotKind::WriteBack).unwrap();
        // Second allocation starts one full page after first (bump rounds up)
        assert_eq!(r2.offset - r1.offset, 4096);
    }

    #[test]
    fn zone_full_returns_error() {
        let mut a = small_alloc();
        a.alloc(4096 * 2, PgprotKind::Uncached).unwrap();
        assert!(a.alloc(1, PgprotKind::Uncached).is_err());
        assert!(a.alloc(4096, PgprotKind::WriteCombine).is_ok());
    }

    #[test]
    fn cross_zone_isolation() {
        // Filling WC zone should not affect UC or WB
        let mut a = small_alloc();
        a.alloc(4096 * 4, PgprotKind::WriteCombine).unwrap();
        assert!(a.alloc(4096, PgprotKind::WriteCombine).is_err());
        assert!(a.alloc(4096, PgprotKind::Uncached).is_ok());
        assert!(a.alloc(4096, PgprotKind::WriteBack).is_ok());
    }

    #[test]
    fn can_write_to_memfd_region() {
        let a = small_alloc();
        // Write to the first byte of each zone and read it back,
        // proving the mmap is live.
        unsafe {
            let uc_ptr = a.base_ptr().add(a.uc.base as usize);
            let wc_ptr = a.base_ptr().add(a.wc.base as usize);
            let wb_ptr = a.base_ptr().add(a.wb.base as usize);

            *uc_ptr = 0xAA;
            *wc_ptr = 0xBB;
            *wb_ptr = 0xCC;

            assert_eq!(*uc_ptr, 0xAA);
            assert_eq!(*wc_ptr, 0xBB);
            assert_eq!(*wb_ptr, 0xCC);
        }
    }

    #[test]
    fn map_host_fd_with_devnull() {
        // We can't map /dev/null with MAP_SHARED, but we can test with
        // another memfd to prove the MAP_FIXED path works.
        let mut a = small_alloc();
        let region = a.alloc(4096, PgprotKind::WriteCombine).unwrap();

        // Create a second memfd, write a magic value, then map it in.
        let name = CString::new("test-host-fd").unwrap();
        let host_fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
        assert!(host_fd >= 0);
        unsafe {
            libc::ftruncate(host_fd, 4096);
            let tmp = libc::mmap(
                ptr::null_mut(),
                4096,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                host_fd,
                0,
            );
            assert_ne!(tmp, libc::MAP_FAILED);
            *(tmp as *mut u8) = 0x42;
            libc::munmap(tmp, 4096);
        }

        // Map the host fd into our SHM region.
        unsafe {
            a.map_host_fd(region.offset, 4096, host_fd).unwrap();
        }

        // Read through the SHM base pointer — should see the magic value.
        unsafe {
            let val = *a.base_ptr().add(region.offset as usize);
            assert_eq!(
                val, 0x42,
                "MAP_FIXED should have overlaid the host fd mapping"
            );
        }

        unsafe { libc::close(host_fd) };
    }
}
