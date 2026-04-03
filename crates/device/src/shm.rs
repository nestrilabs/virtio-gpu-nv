// crates/device/src/shm.rs

use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::ptr;

use crate::error::{DeviceError, Result};

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PgprotKind {
    WriteBack = 0,
    WriteCombine = 1,
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

#[derive(Debug)]
pub struct ShmRegion {
    pub offset: u64,
    pub length: u64,
    pub pgprot: PgprotKind,
}

struct Zone {
    base: u64,
    size: u64,
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

pub struct ZoneConfig {
    pub uc_size: u64,
    pub wc_size: u64,
    pub wb_size: u64,
}

impl ZoneConfig {
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

pub struct ShmAllocator {
    uc: Zone,
    wc: Zone,
    wb: Zone,

    /// Base pointer for MAP_FIXED operations.
    /// Initially points to the memfd mmap (self-owned fallback).
    /// Overridden to the guest memory HVA via set_base_ptr().
    base_ptr: *mut u8,

    /// Self-owned memfd mapping — used as fallback when no external
    /// base pointer is provided (e.g., unit tests).
    memfd: Option<OwnedFd>,
    memfd_ptr: *mut u8,
    memfd_size: u64,

    total_size: u64,
}

unsafe impl Send for ShmAllocator {}
unsafe impl Sync for ShmAllocator {}

impl ShmAllocator {
    pub fn new(cfg: ZoneConfig) -> Self {
        assert_eq!(cfg.uc_size % 4096, 0);
        assert_eq!(cfg.wc_size % 4096, 0);
        assert_eq!(cfg.wb_size % 4096, 0);

        let total = cfg.total();
        assert!(total > 0);

        // Create a memfd as fallback backing (used for tests and
        // before set_base_ptr is called).
        let name = CString::new("virtio-gpu-nv-shm").unwrap();
        let raw_fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
        assert!(
            raw_fd >= 0,
            "memfd_create failed: {}",
            std::io::Error::last_os_error()
        );
        let memfd = unsafe { OwnedFd::from_raw_fd(raw_fd) };

        let ret = unsafe { libc::ftruncate(memfd.as_raw_fd(), total as libc::off_t) };
        assert_eq!(
            ret,
            0,
            "ftruncate failed: {}",
            std::io::Error::last_os_error()
        );

        let memfd_ptr = unsafe {
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
            memfd_ptr,
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
            base_ptr: memfd_ptr as *mut u8,
            memfd: Some(memfd),
            memfd_ptr: memfd_ptr as *mut u8,
            memfd_size: total,
            total_size: total,
        }
    }

    pub fn with_default_zones() -> Self {
        Self::new(ZoneConfig::default_256mib())
    }

    /// Override the base pointer used for MAP_FIXED operations.
    ///
    /// Called by the VMM after guest memory is set up, passing the HVA
    /// of the guest physical address range allocated for the SHM BAR.
    /// After this call, map_host_fd() will mmap directly into guest
    /// memory (visible via EPT/NPT), not into the memfd.
    pub fn set_base_ptr(&mut self, ptr: *mut u8) {
        log::info!(
            "ShmAllocator: base_ptr updated from {:?} to {:?}",
            self.base_ptr,
            ptr
        );
        self.base_ptr = ptr;
    }

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

    /// mmap a host fd into the SHM region at the given offset.
    ///
    /// Uses MAP_FIXED to overlay the host GPU mapping onto the base
    /// pointer region (either guest memory HVA or memfd fallback).
    ///
    /// # Safety
    ///
    /// `host_fd` must be a valid fd that supports mmap.
    /// `offset` and `length` must be within the BAR and page-aligned.
    pub unsafe fn map_host_fd(&self, offset: u64, length: u64, host_fd: RawFd) -> Result<()> {
        let target = unsafe { self.base_ptr.add(offset as usize) as *mut libc::c_void };

        log::info!(
            "SHM map_host_fd: base_ptr={:?} offset=0x{:x} target={:?} len=0x{:x} fd={}",
            self.base_ptr,
            offset,
            target,
            length,
            host_fd
        );

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
        log::info!(
            "SHM map_host_fd: mapped fd={} at offset=0x{:x} length=0x{:x} result={:?}",
            host_fd,
            offset,
            length,
            ptr
        );
        Ok(())
    }

    pub fn memfd_raw(&self) -> RawFd {
        self.memfd.as_ref().map_or(-1, |fd| fd.as_raw_fd())
    }

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
        // Only unmap the memfd mapping, not the guest memory.
        if !self.memfd_ptr.is_null() {
            unsafe {
                libc::munmap(
                    self.memfd_ptr as *mut libc::c_void,
                    self.memfd_size as usize,
                );
            }
            self.memfd_ptr = ptr::null_mut();
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
        assert_eq!(a.uc.base + a.uc.size, a.wc.base);
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
    }

    #[test]
    fn alloc_respects_page_alignment() {
        let mut a = small_alloc();
        let r1 = a.alloc(1, PgprotKind::WriteBack).unwrap();
        let r2 = a.alloc(1, PgprotKind::WriteBack).unwrap();
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
    fn set_base_ptr_changes_target() {
        let mut a = small_alloc();
        let original = a.base_ptr();
        let fake_ptr = 0xDEAD_0000 as *mut u8;
        a.set_base_ptr(fake_ptr);
        assert_eq!(a.base_ptr(), fake_ptr);
        assert_ne!(a.base_ptr(), original);
    }

    #[test]
    fn map_host_fd_with_memfd_fallback() {
        // Tests using the default memfd-backed base_ptr (no set_base_ptr call)
        let mut a = small_alloc();
        let region = a.alloc(4096, PgprotKind::WriteCombine).unwrap();

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

        unsafe {
            a.map_host_fd(region.offset, 4096, host_fd).unwrap();
        }

        unsafe {
            let val = *a.base_ptr().add(region.offset as usize);
            assert_eq!(val, 0x42);
        }

        unsafe { libc::close(host_fd) };
    }
}
