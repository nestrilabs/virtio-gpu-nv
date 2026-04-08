// src/devices/src/virtio/gpu_nv/mmap.rs
//
// handle_mmap — the critical piece that makes steady-state GPU operations run
// at native speed.
//
// Flow:
//   1. Perform mmap() on the host NVIDIA device fd.
//   2. Allocate a guest physical address (GPA) from the MMIO window.
//   3. Register a KVM memory slot: GPA → host virtual address.
//   4. Return GPA + mapping_id to the guest driver.
//
// The guest driver then calls remap_pfn_range() to wire that GPA into the
// requesting process's virtual address space.  From that point on, every
// GPU command write goes directly through EPT to the host physical pages
// the GPU is DMA'ing from — zero VMM involvement.

use std::os::unix::io::AsRawFd;

use kvm_bindings::kvm_userspace_memory_region;

use crate::virtio::gpu_nv::device::GpuMapping;
use crate::virtio::gpu_nv::{
    bytes_of, GpuNv, NvgpuMmapReq, NvgpuMmapResp, NvgpuMsgHdr, NvgpuMunmapReq, NVGPU_MSG_MMAP,
    NVGPU_MSG_MUNMAP,
};

impl GpuNv {
    // ─────────────────────────────────────────────────────────────────────────
    // MMAP
    // ─────────────────────────────────────────────────────────────────────────

    pub(crate) fn handle_mmap(&mut self, req_buf: &[u8]) -> Vec<u8> {
        if req_buf.len() < std::mem::size_of::<NvgpuMmapReq>() {
            return self.error_response(0, -libc::EINVAL);
        }

        let req: NvgpuMmapReq =
            unsafe { std::ptr::read_unaligned(req_buf.as_ptr() as *const NvgpuMmapReq) };

        // ── 1. Lookup host FD ────────────────────────────────────────────────
        let handle = req.hdr.handle;
        let raw_fd = match self.fd_table.get(&handle) {
            Some(f) => f.fd.as_raw_fd(),
            None => return self.error_response(req.hdr.handle, -libc::EBADF),
        };

        // ── 2. mmap on the host ──────────────────────────────────────────────
        //
        // The offset is opaque — it encodes what the NVIDIA driver should map:
        // GPFIFO ring, doorbell BAR0 page, VRAM aperture page, etc.
        // We forward it verbatim; the driver returns the correct host pages.
        let host_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                req.size as libc::size_t,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                raw_fd,
                req.offset as libc::off_t,
            )
        };

        if host_ptr == libc::MAP_FAILED {
            let e = std::io::Error::last_os_error();
            return self.error_response(req.hdr.handle, -e.raw_os_error().unwrap_or(libc::EIO));
        }

        // ── 3. Allocate a guest physical address ─────────────────────────────
        let guest_phys = match self.mmio_alloc.alloc(req.size) {
            Some(addr) => addr,
            None => {
                unsafe { libc::munmap(host_ptr, req.size as libc::size_t) };
                return self.error_response(req.hdr.handle, -libc::ENOMEM);
            }
        };

        // ── 4. Create KVM memory slot ────────────────────────────────────────
        //
        // KVM walks the host page tables, finds the physical frames behind
        // `host_ptr` (whether RAM-backed or device MMIO/WC), and builds EPT
        // entries that map `guest_phys_addr` → those frames.
        // From this moment the guest can access the GPU memory directly.
        let slot = self.next_kvm_slot;
        self.next_kvm_slot += 1;

        let mem_region = kvm_userspace_memory_region {
            slot,
            flags: 0,
            guest_phys_addr: guest_phys,
            memory_size: req.size,
            userspace_addr: host_ptr as u64,
        };

        if let Err(e) = unsafe { self.vm_fd.set_user_memory_region(mem_region) } {
            unsafe { libc::munmap(host_ptr, req.size as libc::size_t) };
            eprintln!("virtio-gpu-nv: KVM slot creation failed: {}", e);
            return self.error_response(req.hdr.handle, -libc::EIO);
        }

        // ── 5. Track the mapping ─────────────────────────────────────────────
        let mapping_id = self.next_mapping_id;
        self.next_mapping_id += 1;

        self.mappings.insert(
            mapping_id,
            GpuMapping {
                host_ptr,
                size: req.size,
                guest_phys_addr: guest_phys,
                kvm_slot: slot,
                host_fd_handle: req.hdr.handle,
            },
        );

        let handle = req.hdr.handle;
        if let Some(fd) = self.fd_table.get_mut(&handle) {
            fd.mapping_ids.push(mapping_id);
        }

        // ── 6. Return GPA to guest driver ────────────────────────────────────
        bytes_of(&NvgpuMmapResp {
            hdr: NvgpuMsgHdr {
                msg_type: NVGPU_MSG_MMAP,
                handle: req.hdr.handle,
                status: 0,
                padding: 0,
            },
            guest_phys_addr: guest_phys,
            size: req.size,
            mapping_id,
            padding: 0,
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MUNMAP
    // ─────────────────────────────────────────────────────────────────────────

    pub(crate) fn handle_munmap(&mut self, req_buf: &[u8]) -> Vec<u8> {
        if req_buf.len() < std::mem::size_of::<NvgpuMunmapReq>() {
            return self.error_response(0, -libc::EINVAL);
        }

        let req: NvgpuMunmapReq =
            unsafe { std::ptr::read_unaligned(req_buf.as_ptr() as *const NvgpuMunmapReq) };

        self.destroy_mapping(req.mapping_id);

        bytes_of(&NvgpuMsgHdr {
            msg_type: NVGPU_MSG_MUNMAP,
            handle: req.hdr.handle,
            status: 0,
            padding: 0,
        })
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal: tear down one KVM memory slot
    // ─────────────────────────────────────────────────────────────────────────

    pub(crate) fn destroy_mapping(&mut self, mapping_id: u32) {
        if let Some(mapping) = self.mappings.remove(&mapping_id) {
            // Remove the KVM slot by setting memory_size = 0.
            let remove = kvm_userspace_memory_region {
                slot: mapping.kvm_slot,
                flags: 0,
                guest_phys_addr: mapping.guest_phys_addr,
                memory_size: 0,
                userspace_addr: 0,
            };
            let _ = unsafe { self.vm_fd.set_user_memory_region(remove) };

            // Unmap the host virtual address range.
            unsafe {
                libc::munmap(mapping.host_ptr, mapping.size as libc::size_t);
            }
        }
    }
}
