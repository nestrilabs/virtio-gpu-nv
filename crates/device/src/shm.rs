// crates/device/src/shm.rs
//
// SHM BAR region allocator — with per-cache-type zone partitioning.
//
// --- Why cache types matter ---
//
// NVIDIA GPU mappings require specific CPU caching attributes:
//
//   UC  (uncached)        — GPU control registers, doorbells, semaphores.
//                           Writes must be visible immediately; no coalescing.
//   WC  (write-combining) — VRAM framebuffer BAR, push buffers.
//                           Burst writes are efficient; reads are slow.
//   WB  (write-back)      — System memory mapped for GPU DMA (e.g. after
//                           cuMemHostAlloc with cudaHostAllocMapped).
//
// A single contiguous BAR region cannot be mapped with mixed pgprot because
// the x86 MTRRs and PAT must agree with the EPT mapping the VMM sets up.
// If QEMU/KVM maps a BAR as WB in the EPT and the guest PAT tries to
// downgrade a page to UC, the hardware enforces the stricter of the two —
// which is UC — but only if the MTRR/EPT is already UC or WC.  Mapping
// a UC device register as WB in the EPT will cause data corruption on
// real hardware.
//
// --- Solution: three fixed zones ---
//
// We partition the BAR into three contiguous zones at construction time:
//
//   [0 .. uc_end)         — UC zone  (smallest: registers are few, small)
//   [uc_end .. wc_end)    — WC zone  (largest: framebuffer can be GiBs)
//   [wc_end .. bar_size)  — WB zone  (medium: DMA buffers)
//
// Each zone has its own bump cursor.  The VMM must map these three sub-ranges
// into the EPT with the matching memory type (UC/WC/WB).  The guest driver's
// nv_mmap() reads the pgprot from ioctl_resp and uses pgprot_noncached /
// pgprot_writecombine / PAGE_KERNEL accordingly — but that only works
// correctly if the EPT entry already agrees.
//
// Default zone split (tunable at construction): 4 MiB UC, 128 MiB WC, rest WB.
// These are generous defaults; real NVIDIA register BARs are < 1 MiB.

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

/// BAR zone configuration passed to `ShmAllocator::new`.
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

/// Partitioned bump allocator for the SHM BAR.
///
/// Each allocation is tagged with a `PgprotKind` and comes from the
/// corresponding zone.  The VMM is responsible for mapping each zone into
/// the EPT with the matching memory type before the guest boots.
pub struct ShmAllocator {
    uc: Zone,
    wc: Zone,
    wb: Zone,
}

impl ShmAllocator {
    pub fn new(cfg: ZoneConfig) -> Self {
        assert!(cfg.uc_size % 4096 == 0);
        assert!(cfg.wc_size % 4096 == 0);
        assert!(cfg.wb_size % 4096 == 0);
        let uc_base = 0;
        let wc_base = cfg.uc_size;
        let wb_base = cfg.uc_size + cfg.wc_size;
        Self {
            uc: Zone::new(uc_base, cfg.uc_size),
            wc: Zone::new(wc_base, cfg.wc_size),
            wb: Zone::new(wb_base, cfg.wb_size),
        }
    }

    /// Convenience constructor using the default 256 MiB split.
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

    /// Byte offset of the start of the UC zone (for EPT setup).
    pub fn uc_zone_offset(&self) -> u64 {
        self.uc.base
    }
    /// Byte offset of the start of the WC zone (for EPT setup).
    pub fn wc_zone_offset(&self) -> u64 {
        self.wc.base
    }
    /// Byte offset of the start of the WB zone (for EPT setup).
    pub fn wb_zone_offset(&self) -> u64 {
        self.wb.base
    }

    /// Total BAR size = UC + WC + WB zone sizes.
    pub fn total_size(&self) -> u64 {
        self.uc.size + self.wc.size + self.wb.size
    }
}

fn align_up(v: u64, align: u64) -> u64 {
    (v + align - 1) & !(align - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_alloc() -> ShmAllocator {
        ShmAllocator::new(ZoneConfig {
            uc_size: 4096 * 2,
            wc_size: 4096 * 4,
            wb_size: 4096 * 2,
        })
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
        a.alloc(4096 * 2, PgprotKind::Uncached).unwrap(); // fills the 2-page UC zone
        assert!(a.alloc(1, PgprotKind::Uncached).is_err()); // no room left
                                                            // Other zones still work
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
}
