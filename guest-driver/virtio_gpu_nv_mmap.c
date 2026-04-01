// SPDX-License-Identifier: Apache-2.0
/*
 * virtio_gpu_nv_mmap.c — mmap file operation
 *
 * Called by userspace after a mapping ioctl (e.g. NV_ESC_RM_MAP_MEMORY).
 * The backend has already:
 *   1. Issued the host ioctl.
 *   2. mmap'd the resulting host mapping into the SHM BAR at some offset.
 *   3. Returned (shm_offset, shm_length, pgprot) in the ioctl response.
 *
 * The NVIDIA user-mode driver then calls mmap(fd, offset=shm_offset) to
 * get the GPU mapping into its address space.  We satisfy that here by
 * calling remap_pfn_range() to map the corresponding SHM BAR PFNs.
 *
 * Phase 1: The backend never returns shm_offset != 0 (no mapping ioctls
 * yet), so this path is exercised in Phase 3.  The code is written now
 * so the infrastructure is in place.
 */

#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/virtio.h>

#include "virtio_gpu_nv.h"
#include "virtio_gpu_nv_priv.h"

/*
 * PFN of the first byte of the SHM BAR in the guest physical address space.
 *
 * Filled in during virtio_probe when the VMM exposes the BAR via a
 * virtio memory region or a KVM memslot.  Phase 1: set to 0 (unused).
 *
 * Phase 3 will read this from virtio config space:
 *   virtio_cread(vdev, struct nv_config, shm_bar_gpa, &ndev->shm_bar_gpa);
 */
static unsigned long g_shm_bar_pfn; /* guest PFN of SHM BAR base */

/* Called from probe once we know the GPA of the SHM BAR. */
void nv_set_shm_bar_pfn(unsigned long pfn) { g_shm_bar_pfn = pfn; }

/* -------------------------------------------------------------------------
 * nv_mmap
 * ---------------------------------------------------------------------- */

int nv_mmap(struct file *filp, struct vm_area_struct *vma) {
  unsigned long size = vma->vm_end - vma->vm_start;
  unsigned long shm_off = vma->vm_pgoff << PAGE_SHIFT;
  unsigned long pfn;
  pgprot_t pgprot;
  int ret;

  /*
   * vma->vm_pgoff encodes the SHM BAR offset (in pages) returned by
   * the backend in ioctl_resp::shm_offset.  The NVIDIA user-mode
   * driver passes this as the `offset` argument to mmap(2).
   *
   * We recover the pgprot from the VMA flags set by the user-mode
   * driver.  NVIDIA's mmap(2) call sets PROT_READ|PROT_WRITE and
   * relies on the driver to enforce the correct caching policy.
   *
   * For Phase 1, g_shm_bar_pfn is 0, so any mmap attempt will fail
   * gracefully.  Phase 3 sets g_shm_bar_pfn from config space.
   */

  if (!g_shm_bar_pfn) {
    pr_warn_once("nv_mmap: SHM BAR not configured\n");
    return -ENODEV;
  }

  pfn = g_shm_bar_pfn + (shm_off >> PAGE_SHIFT);

  /*
   * Caching policy.
   *
   * The backend encodes the desired pgprot in ioctl_resp::pgprot:
   *   0 = write-back   (normal GPU framebuffer reads)
   *   1 = write-combining (GPU MMIO bars, framebuffer writes)
   *   2 = uncached      (device registers, doorbells)
   *
   * For Phase 1/2 we default to write-combining which is correct for
   * most NVIDIA MMIO regions.  Phase 3 will store the pgprot hint in
   * nv_file_ctx and read it here.
   */
  pgprot = pgprot_writecombine(vma->vm_page_prot);

  /* Mark the VMA as non-cacheable so the kernel doesn't try to swap it. */
  vma->vm_flags |= VM_IO | VM_DONTEXPAND | VM_DONTDUMP;
  vma->vm_page_prot = pgprot;

  ret = remap_pfn_range(vma, vma->vm_start, pfn, size, pgprot);
  if (ret)
    pr_err("nv_mmap: remap_pfn_range failed: %d\n", ret);

  return ret;
}
