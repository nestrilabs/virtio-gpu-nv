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
#include <linux/list.h>
#include <linux/mm.h>
#include <linux/spinlock.h>
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
  struct nv_file_ctx *ctx = filp->private_data;
  unsigned long size = vma->vm_end - vma->vm_start;
  unsigned long shm_off = vma->vm_pgoff << PAGE_SHIFT;
  struct nv_mapping_info *mi, *found = NULL;
  unsigned long pfn;
  pgprot_t pgprot;
  int ret;

  if (!g_shm_bar_pfn) {
    pr_warn_once("nv_mmap: SHM BAR not configured\n");
    return -ENODEV;
  }

  /* Look up the mapping info by SHM offset. */
  spin_lock(&ctx->mappings_lock);
  list_for_each_entry(mi, &ctx->mappings, list) {
    if (mi->shm_offset == shm_off) {
      found = mi;
      break;
    }
  }
  spin_unlock(&ctx->mappings_lock);

  if (!found) {
    pr_err("nv_mmap: no mapping info for offset 0x%lx\n", shm_off);
    return -EINVAL;
  }

  if (size > found->shm_length) {
    pr_err("nv_mmap: requested size 0x%lx exceeds mapping length 0x%llx\n",
           size, found->shm_length);
    return -EINVAL;
  }

  pfn = g_shm_bar_pfn + (shm_off >> PAGE_SHIFT);

  /* Select pgprot based on the backend's caching type hint. */
  switch (found->pgprot) {
  case 0: /* WB — write-back */
    pgprot = vma->vm_page_prot;
    break;
  case 1: /* WC — write-combining */
    pgprot = pgprot_writecombine(vma->vm_page_prot);
    break;
  case 2: /* UC — uncached */
    pgprot = pgprot_noncached(vma->vm_page_prot);
    break;
  default:
    pr_warn("nv_mmap: unknown pgprot %u, defaulting to UC\n", found->pgprot);
    pgprot = pgprot_noncached(vma->vm_page_prot);
    break;
  }

  vm_flags_set(vma, VM_IO | VM_DONTEXPAND | VM_DONTDUMP);
  vma->vm_page_prot = pgprot;

  ret = remap_pfn_range(vma, vma->vm_start, pfn, size, pgprot);
  if (ret)
    pr_err("nv_mmap: remap_pfn_range failed: %d\n", ret);

  return ret;
}
