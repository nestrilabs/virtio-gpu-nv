/* SPDX-License-Identifier: GPL-2.0 */
/*
 * nvgpu_v1v2_rewrites.h — V1→V2 ioctl rewrite table for RM_CONTROL
 *
 * Many RM_CONTROL commands have two variants:
 *   V1: nested params contain a userspace pointer to a data buffer
 *   V2: nested params contain inline data (no second-level pointer)
 *
 * The guest driver cannot forward V1 to the VMM because the VMM cannot
 * dereference guest userspace pointers inside nested params.  Instead,
 * we rewrite V1 → V2 before sending, and reverse the transform on the
 * response path.
 *
 * Two patterns:
 *
 * GET_CAPS V1:  {u32 capsTblSize, pad(4), NvP64 capsTbl}     (16 bytes)
 *   - userptr at offset 8, capsTblSize is byte count
 *   - no prefix to copy into V2 (v1_copy_prefix = 0)
 *
 * GET_INFO V1:  {u32 listSize, pad(4), NvP64 list}           (16 bytes)
 *   - userptr at offset 8, listSize is item count (each item 8 bytes)
 *   - copy listSize (4 bytes) into V2 at offset 0
 *
 * CE GET_CAPS V1: {u32 ceEngineType, u32 capsTblSize, NvP64 capsTbl} (24 bytes)
 *   - userptr at offset 16, ceEngineType prefix (4 bytes) into V2
 */

#ifndef NVGPU_V1V2_REWRITES_H
#define NVGPU_V1V2_REWRITES_H

#include <linux/types.h>

struct nvgpu_v1v2_entry {
  u32 v1_cmd;            /* RM_CONTROL cmd that carries V1 layout      */
  u32 v2_cmd;            /* replacement cmd with inline V2 layout      */
  u32 v2_size;           /* sizeof the V2 nested params struct         */
  u32 v1_userptr_offset; /* byte offset of NvP64 in V1 nested params  */
  u32 v1_copy_prefix;    /* bytes to copy from V1 start → V2 start    */
  u32 v2_data_offset;    /* offset where result data begins in V2     */
  u32 v2_data_size;      /* max bytes of result data to copy back     */
  bool info_style;       /* true = listSize is item count (×8 bytes)  */
};

/*
 * Sizing notes (V2 structs with padding/alignment):
 *
 * GR_GET_CAPS V2:     {u8[23], pad(1), GR_ROUTE_INFO(16), NvBool(4), pad(4)} =
 * 48 GR_GET_INFO V2:     {u32(4), GR_INFO[59](472), pad(4), GR_ROUTE_INFO(16)}
 * = 496 MSENC_GET_CAPS V2:  {u8[6], pad(2), u32} = 12 NVJPG_GET_CAPS V2:
 * {u8[9], pad(3), u32}                                   = 16 CE_GET_CAPS V2:
 * {u32, u8[2], pad(2)}                                   = 8 FB_GET_INFO V2:
 * {u32(4), FB_INFO[128](1024)}                           = 1028 GPU_GET_INFO
 * V2:    {u32(4), GPU_INFO[70](560)}                            = 564
 * BUS_GET_INFO V2:    {u32(4), BUS_INFO[52](416)}                            =
 * 420 BIOS_GET_INFO V2:   {u32(4), BIOS_INFO[15](120)} = 124
 */

static const struct nvgpu_v1v2_entry nvgpu_v1v2_table[] = {

    /* ── GET_CAPS: V1 = {u32 capsTblSize, pad, NvP64 capsTbl} ──────── */
    /*              v1_cmd      v2_cmd     v2sz  ptr  pfx  d_off d_sz info */

    /* FB_GET_CAPS */
    {0x00801301, 0x00801307, 3, 8, 0, 0, 3, false},
    /* HOST_GET_CAPS */
    {0x00801401, 0x00801402, 3, 8, 0, 0, 3, false},
    /* FIFO_GET_CAPS */
    {0x00801701, 0x00801713, 2, 8, 0, 0, 2, false},
    /* GR_GET_CAPS */
    {0x00801102, 0x00801109, 48, 8, 0, 0, 23, false},
    /* MSENC_GET_CAPS */
    {0x00801b01, 0x00801b02, 12, 8, 0, 0, 6, false},
    /* NVJPG_GET_CAPS */
    {0x00801f01, 0x00801f02, 16, 8, 0, 0, 9, false},
    /* BSP_GET_CAPS */
    {0x00801c01, 0x00801c02, 8, 8, 0, 0, 8, false},

    /* CE_GET_CAPS: V1 = {u32 ceEngineType, u32 capsTblSize, NvP64} */
    {0x20802a01, 0x20802a03, 8, 16, 4, 4, 2, false},

    /* ── GET_INFO: V1 = {u32 listSize, pad, NvP64 list} ────────────── */

    /* GR_GET_INFO (device) */
    {0x00801104, 0x00801110, 496, 8, 4, 4, 472, true},
    /* FB_GET_INFO (subdevice) */
    {0x20801301, 0x20801303, 1028, 8, 4, 4, 1024, true},
    /* GR_GET_INFO (subdevice) */
    {0x20801201, 0x20801228, 496, 8, 4, 4, 472, true},
    /* GPU_GET_INFO */
    {0x20800101, 0x20800102, 564, 8, 4, 4, 560, true},
    /* BUS_GET_INFO */
    {0x20801802, 0x20801823, 420, 8, 4, 4, 416, true},
    /* BIOS_GET_INFO */
    {0x20800802, 0x20800810, 124, 8, 4, 4, 120, true},
};

#define NVGPU_V1V2_TABLE_SIZE ARRAY_SIZE(nvgpu_v1v2_table)

static inline const struct nvgpu_v1v2_entry *nvgpu_find_v1v2_rewrite(u32 cmd) {
  int i;
  for (i = 0; i < (int)NVGPU_V1V2_TABLE_SIZE; i++) {
    if (nvgpu_v1v2_table[i].v1_cmd == cmd)
      return &nvgpu_v1v2_table[i];
  }
  return NULL;
}

#endif /* NVGPU_V1V2_REWRITES_H */
