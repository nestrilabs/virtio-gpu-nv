"""
nvgpu_gen.py — Generate virtio-gpu-nv guest driver tables from
               NVIDIA open-gpu-kernel-modules source tree.

Usage:
    python3 nvgpu_gen.py --src /path/to/open-gpu-kernel-modules --out /path/to/output/ --phase <all|rmalloc|v1v2>

Phase 1: rmalloc classes (hClass → param size)
Phase 2: V1→V2 rewrite table
Phase 3: multi-pointer detection (TODO)
"""

import argparse
import os
import re
import sys
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Phase 1: Parse resource_list.h → (external_class, alloc_param_info)
# ---------------------------------------------------------------------------

@dataclass
class RSEntry:
    external_class: str  # e.g. "NV01_DEVICE_0"
    internal_class: str  # e.g. "Device"
    alloc_param_info: str  # "RS_NONE", "RS_REQUIRED(Type)", "RS_OPTIONAL(Type)"
    alloc_param_type: Optional[str] = None  # Extracted type name, or None for RS_NONE
    alloc_param_required: Optional[bool] = None  # True=REQUIRED, False=OPTIONAL, None=NONE


def parse_resource_list(src_root: Path) -> list[RSEntry]:
    """
    Parse resource_list.h for RS_ENTRY blocks.

    Each RS_ENTRY has 8 fields. We care about fields 0 (External Class)
    and 4 (Alloc Param Info).
    """
    rl_path = src_root / "src" / "nvidia" / "src" / "kernel" / "rmapi" / "resource_list.h"
    if not rl_path.exists():
        print(f"ERROR: {rl_path} not found", file=sys.stderr)
        sys.exit(1)

    text = rl_path.read_text()

    # Strip C comments (both // and /* */)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'//[^\n]*', '', text)

    # Find all RS_ENTRY(...) blocks
    # These can span many lines, so we match the balanced parens
    entries = []
    for m in re.finditer(r'RS_ENTRY\s*\(', text):
        start = m.end()
        depth = 1
        pos = start
        while depth > 0 and pos < len(text):
            if text[pos] == '(':
                depth += 1
            elif text[pos] == ')':
                depth -= 1
            pos += 1
        body = text[start:pos - 1]

        # Split into fields by top-level commas (not inside nested parens)
        fields = split_top_level_commas(body)
        if len(fields) < 5:
            continue

        external_class = fields[0].strip()
        internal_class = fields[1].strip()
        alloc_param_info = fields[4].strip()

        entry = RSEntry(
            external_class=external_class,
            internal_class=internal_class,
            alloc_param_info=alloc_param_info,
        )

        # Parse alloc param info
        none_match = re.match(r'RS_NONE', alloc_param_info)
        req_match = re.match(r'RS_REQUIRED\(\s*(\w+)\s*\)', alloc_param_info)
        opt_match = re.match(r'RS_OPTIONAL\(\s*(\w+)\s*\)', alloc_param_info)

        if none_match:
            entry.alloc_param_type = None
            entry.alloc_param_required = None
        elif req_match:
            entry.alloc_param_type = req_match.group(1)
            entry.alloc_param_required = True
        elif opt_match:
            entry.alloc_param_type = opt_match.group(1)
            entry.alloc_param_required = False
        else:
            print(f"  WARN: Unknown alloc param info format: {alloc_param_info!r} "
                  f"for {external_class}", file=sys.stderr)

        entries.append(entry)

    print(f"Parsed {len(entries)} RS_ENTRY blocks from resource_list.h")
    return entries


def split_top_level_commas(s: str) -> list[str]:
    """Split string by commas, respecting nested parentheses."""
    parts = []
    depth = 0
    current = []
    for ch in s:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current))
    return parts


# ---------------------------------------------------------------------------
# Resolve External Class names → numeric hClass values
# ---------------------------------------------------------------------------

def find_hclass_defines(src_root: Path, class_names: set[str]) -> dict[str, int]:
    """
    Search NVIDIA headers for #define CLASS_NAME <value> patterns.

    The class numeric defines live in various places:
      - src/common/sdk/nvidia/inc/class/cl*.h
      - src/common/sdk/nvidia/inc/nvos.h
      - src/common/sdk/nvidia/inc/class/*.h
      - kernel-open/common/inc/nv-kernel-rmapi-ops.h
    """
    search_dirs = [
        src_root / "src" / "common" / "sdk" / "nvidia" / "inc",
        src_root / "kernel-open" / "common" / "inc",
    ]

    remaining = set(class_names)
    defines = {}

    # Pattern: #define NV01_DEVICE_0 (0x0080)  or  #define NV01_DEVICE_0 0x0080U
    define_re = re.compile(
        r'^\s*#\s*define\s+(\w+)\s+\(?\s*(0x[0-9a-fA-F]+)[Uu]?\s*\)?',
        re.MULTILINE
    )

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for hfile in search_dir.rglob("*.h"):
            if not remaining:
                break
            try:
                content = hfile.read_text(errors='replace')
            except Exception:
                continue
            for m in define_re.finditer(content):
                name, value = m.group(1), m.group(2)
                if name in remaining:
                    defines[name] = int(value, 16)
                    remaining.discard(name)

    if remaining:
        print(f"  WARN: Could not resolve {len(remaining)} class names to numeric values:",
              file=sys.stderr)
        for name in sorted(remaining):
            print(f"    {name}", file=sys.stderr)

    return defines


# ---------------------------------------------------------------------------
# Resolve alloc param type names → sizeof via compiled C probe
# ---------------------------------------------------------------------------

def find_header_for_type(src_root: Path, type_name: str) -> Optional[Path]:
    """Find which header defines a given struct/typedef."""
    search_dirs = [
        src_root / "src" / "common" / "sdk" / "nvidia" / "inc",
        src_root / "kernel-open" / "common" / "inc",
    ]

    # Match "typedef struct ... { ... } TYPE_NAME;"
    # or just "typedef struct TYPE_NAME {"
    patterns = [
        re.compile(rf'\b{re.escape(type_name)}\b'),
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for hfile in search_dir.rglob("*.h"):
            try:
                content = hfile.read_text(errors='replace')
            except Exception:
                continue
            for pat in patterns:
                if pat.search(content):
                    return hfile
    return None


def resolve_param_sizes_via_compile(
        src_root: Path,
        entries: list[RSEntry],
        hclass_map: dict[str, int],
) -> dict[int, tuple[int, str, str, bool]]:
    """
    Compile a tiny C program to get sizeof() for each alloc param type.

    Returns: {hclass_value: (sizeof, external_class_name, param_type_name, is_required)}
    """
    # Collect unique param types that need sizeof
    type_to_entries: dict[str, list[RSEntry]] = {}
    for entry in entries:
        if entry.alloc_param_type and entry.alloc_param_type != "NvHandle":
            type_to_entries.setdefault(entry.alloc_param_type, []).append(entry)

    # Also handle NvHandle specially (it's just u32 = 4 bytes, but let's verify)
    nvhandle_entries = [e for e in entries if e.alloc_param_type == "NvHandle"]

    # Find headers for each type
    type_headers: dict[str, Path] = {}
    for type_name in type_to_entries:
        header = find_header_for_type(src_root, type_name)
        if header:
            type_headers[type_name] = header
        else:
            print(f"  WARN: Cannot find header for type {type_name}", file=sys.stderr)

    # Build include paths
    include_dirs = [
        src_root / "src" / "common" / "sdk" / "nvidia" / "inc",
        src_root / "src" / "common" / "sdk" / "nvidia" / "inc" / "class",
        src_root / "src" / "common" / "sdk" / "nvidia" / "inc" / "ctrl",
        src_root / "kernel-open" / "common" / "inc",
        src_root / "src" / "common" / "inc",
        src_root / "src" / "nvidia" / "inc",
        src_root / "src" / "nvidia" / "arch" / "nvalloc" / "unix" / "include",
    ]

    # Generate the C probe program
    lines = [
        '#include <stdio.h>',
        '#include <stddef.h>',
        '',
        '/* NVIDIA basic types */',
        '#include "nvtypes.h"',
        '#include "nvos.h"',
        '#include "nv_escape.h"',
        '',
    ]

    # Include each discovered header
    included = set()
    for type_name, header in sorted(type_headers.items()):
        # Use path relative to one of the include dirs
        for inc_dir in include_dirs:
            try:
                rel = header.relative_to(inc_dir)
                inc_str = str(rel)
                if inc_str not in included:
                    lines.append(f'#include "{inc_str}"')
                    included.add(inc_str)
                break
            except ValueError:
                continue

    lines.extend([
        '',
        'int main(void) {',
    ])

    # NvHandle entries
    for entry in nvhandle_entries:
        if entry.external_class in hclass_map:
            hval = hclass_map[entry.external_class]
            lines.append(f'    printf("0x%04x %zu %s NvHandle\\n", '
                         f'0x{hval:04x}, sizeof(NvHandle), "{entry.external_class}");')

    # Struct entries
    for type_name, header in sorted(type_headers.items()):
        for entry in type_to_entries[type_name]:
            if entry.external_class in hclass_map:
                hval = hclass_map[entry.external_class]
                lines.append(
                    f'    printf("0x%04x %zu %s %s\\n", '
                    f'0x{hval:04x}, sizeof({type_name}), '
                    f'"{entry.external_class}", "{type_name}");'
                )

    lines.extend([
        '    return 0;',
        '}',
    ])

    probe_src = '\n'.join(lines) + '\n'

    # Write, compile, run
    with tempfile.TemporaryDirectory(prefix="nvgpu_gen_") as tmpdir:
        src_file = os.path.join(tmpdir, "probe.c")
        bin_file = os.path.join(tmpdir, "probe")

        with open(src_file, 'w') as f:
            f.write(probe_src)

        # Also save a copy for debugging
        debug_copy = Path("nvgpu_gen_probe.c")
        debug_copy.write_text(probe_src)
        print(f"  Wrote probe source to {debug_copy} for debugging")

        inc_flags = []
        for d in include_dirs:
            if d.exists():
                inc_flags.extend(["-I", str(d)])

        cmd = ["gcc", "-o", bin_file, src_file] + inc_flags + [
            "-Wno-all",
            "-fsigned-char",
            "-D__packed=__attribute__((packed))",
            "-DNVRM",
            "-DNV_KERNEL_INTERFACE_LAYER",
        ]

        print(f"  Compiling probe: {' '.join(cmd[:6])} ...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: Probe compilation failed:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            # Try to salvage what we can — return empty and fall back to manual
            return {}

        result = subprocess.run([bin_file], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: Probe execution failed", file=sys.stderr)
            return {}

    # Parse output: "0xHHHH SIZE CLASSNAME TYPENAME"
    sizes = {}
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split()
        hval = int(parts[0], 16)
        size = int(parts[1])
        class_name = parts[2]
        type_name = parts[3] if len(parts) > 3 else "NvHandle"

        # Determine if required
        entry = next((e for e in entries if e.external_class == class_name), None)
        is_required = entry.alloc_param_required if entry else None

        sizes[hval] = (size, class_name, type_name, is_required)

    # Add RS_NONE entries (size = 0)
    for entry in entries:
        if entry.alloc_param_type is None and entry.external_class in hclass_map:
            hval = hclass_map[entry.external_class]
            if hval not in sizes:
                sizes[hval] = (0, entry.external_class, "none", None)

    return sizes


# ---------------------------------------------------------------------------
# Emit nvgpu_rmalloc_classes.h
# ---------------------------------------------------------------------------

def emit_rmalloc_header(
        sizes: dict[int, tuple[int, str, str, bool]],
        out_dir: Path,
        driver_version: str,
):
    lines = [
        '/* SPDX-License-Identifier: GPL-2.0 */',
        '/*',
        f' * nvgpu_rmalloc_classes.h — auto-generated from NVIDIA driver {driver_version}',
        ' *',
        ' * Generated by nvgpu_gen.py — DO NOT EDIT MANUALLY',
        ' *',
        ' * hClass → pAllocParms size lookup for RM_ALLOC.',
        ' * When paramsSize == 0 but pAllocParms != NULL, the host RM driver',
        ' * determines the param size from hClass internally. The guest driver',
        ' * must know how many bytes to copy_from_user before forwarding.',
        ' */',
        '',
        '#ifndef NVGPU_RMALLOC_CLASSES_H',
        '#define NVGPU_RMALLOC_CLASSES_H',
        '',
        '#include <linux/types.h>',
        '',
        '#define NVGPU_RMALLOC_FALLBACK_SIZE 512',
        '',
        'static inline u32 nvgpu_rmalloc_class_param_size(u32 hClass)',
        '{',
        '    switch (hClass) {',
    ]

    # Group by category for readability
    # Sort by hClass value
    sorted_entries = sorted(sizes.items())

    # Separate into: no-params (size=0), has-params
    no_params = [(h, v) for h, v in sorted_entries if v[0] == 0]
    has_params = [(h, v) for h, v in sorted_entries if v[0] > 0]

    if has_params:
        lines.append('')
        lines.append('    /* ── Classes with alloc params ── */')
        for hval, (size, class_name, type_name, is_required) in has_params:
            req_str = "required" if is_required else "optional" if is_required is False else ""
            lines.append(f'    case 0x{hval:04x}: return {size:>4};  '
                         f'/* {class_name} ({type_name}) {req_str} */'.rstrip())

    if no_params:
        lines.append('')
        lines.append('    /* ── No-params classes (RS_NONE) ── */')
        for hval, (size, class_name, type_name, _) in no_params:
            lines.append(f'    case 0x{hval:04x}: return    0;  /* {class_name} */')

    lines.extend([
        '',
        '    /* ── Unknown → conservative fallback ── */',
        '    default: return NVGPU_RMALLOC_FALLBACK_SIZE;',
        '    }',
        '}',
        '',
        '#endif /* NVGPU_RMALLOC_CLASSES_H */',
        '',
    ])

    out_path = out_dir / "nvgpu_rmalloc_classes.h"
    out_path.write_text('\n'.join(lines))
    print(f"Wrote {out_path} ({len(has_params)} param classes, {len(no_params)} no-param classes)")


# ---------------------------------------------------------------------------
# Phase 2: V1→V2 RM_CONTROL rewrite detection
# ---------------------------------------------------------------------------

@dataclass
class V1V2Candidate:
    """A detected V1/V2 command pair."""
    v1_cmd_name: str  # e.g. "NV2080_CTRL_CMD_GPU_GET_INFO"
    v2_cmd_name: str  # e.g. "NV2080_CTRL_CMD_GPU_GET_INFO_V2"
    v1_cmd_value: int  # e.g. 0x20800101
    v2_cmd_value: int  # e.g. 0x20800102
    v1_params_type: str  # e.g. "NV2080_CTRL_GPU_GET_INFO_PARAMS"
    v2_params_type: str  # e.g. "NV2080_CTRL_GPU_GET_INFO_V2_PARAMS"
    header_path: Path  # source header file
    # Filled in by C probe:
    v1_size: int = 0
    v2_size: int = 0
    v1_userptr_offset: int = 0
    v1_prefix_size: int = 0  # bytes before NvP64 in V1
    v2_data_offset: int = 0
    v2_data_size: int = 0
    info_style: bool = False  # True = list count × item_size, False = byte count


def find_v1v2_pairs(src_root: Path) -> tuple[list[V1V2Candidate], dict[str, tuple[str, Path]], dict[str, str]]:
    """
    Scan ctrl*.h headers for V1/V2 command pairs.
    """
    ctrl_dirs = [
        src_root / "src" / "common" / "sdk" / "nvidia" / "inc" / "ctrl",
    ]

    cmd_define_re = re.compile(
        r'^\s*#\s*define\s+(NV\w+_CTRL_CMD_\w+)\s+\(?(0x[0-9a-fA-F]+)U?\)?\s*',
        re.MULTILINE
    )

    struct_re = re.compile(
        r'typedef\s+struct\s+(\w+)\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}\s*(\w+)\s*;',
        re.DOTALL
    )

    # Also catch simple typedef aliases:
    #   typedef EXISTING_TYPE NEW_TYPE;
    # but NOT typedef struct, typedef enum, etc.
    alias_re = re.compile(
        r'^\s*typedef\s+(\w+)\s+(\w+)\s*;',
        re.MULTILINE
    )

    all_cmds: dict[str, tuple[int, Path]] = {}
    all_structs: dict[str, tuple[str, Path]] = {}
    all_aliases: dict[str, str] = {}  # alias_name → original_name

    for ctrl_dir in ctrl_dirs:
        if not ctrl_dir.exists():
            continue
        for hfile in ctrl_dir.rglob("*.h"):
            try:
                content = hfile.read_text(errors='replace')
            except Exception:
                continue

            for m in cmd_define_re.finditer(content):
                name, value = m.group(1), m.group(2)
                all_cmds[name] = (int(value, 16), hfile)

            for m in struct_re.finditer(content):
                type_name = m.group(3)
                body = m.group(2)
                all_structs[type_name] = (body, hfile)

            for m in alias_re.finditer(content):
                original, alias = m.group(1), m.group(2)
                # Skip "typedef struct X { } X;" — those are caught by struct_re
                # Skip "typedef enum ..." — original won't be a PARAMS type
                if original in all_structs or original in all_aliases:
                    all_aliases[alias] = original

    # Second pass: resolve aliases that reference other aliases or structs
    # found in later files
    for ctrl_dir in ctrl_dirs:
        if not ctrl_dir.exists():
            continue
        for hfile in ctrl_dir.rglob("*.h"):
            try:
                content = hfile.read_text(errors='replace')
            except Exception:
                continue
            for m in alias_re.finditer(content):
                original, alias = m.group(1), m.group(2)
                if alias not in all_structs and alias not in all_aliases:
                    if original in all_structs or original in all_aliases:
                        all_aliases[alias] = original

    print(f"  Found {len(all_cmds)} command defines, {len(all_structs)} struct definitions, "
          f"{len(all_aliases)} type aliases")

    def resolve_struct(type_name: str) -> tuple[str, str, Path] | None:
        """Resolve a type name to its struct body, following aliases."""
        visited = set()
        current = type_name
        while current not in all_structs:
            if current in all_aliases and current not in visited:
                visited.add(current)
                current = all_aliases[current]
            else:
                return None
        body, path = all_structs[current]
        return current, body, path

    # Find V1/V2 pairs
    candidates = []
    for cmd_name, (cmd_value, cmd_path) in sorted(all_cmds.items()):
        if cmd_name.endswith('_V2'):
            continue

        v2_name = cmd_name + '_V2'
        if v2_name not in all_cmds:
            continue

        v2_value, v2_path = all_cmds[v2_name]

        # Derive V1 params type name
        v1_params = cmd_name.replace('_CTRL_CMD_', '_CTRL_') + '_PARAMS'

        # Resolve V1 params (may be an alias)
        v1_resolved = resolve_struct(v1_params)
        if v1_resolved is None:
            continue
        v1_real_type, v1_body, _ = v1_resolved

        if 'NvP64' not in v1_body:
            continue

        # Try multiple V2 params naming patterns
        v2_params_candidates = [
            cmd_name.replace('_CTRL_CMD_', '_CTRL_') + '_V2_PARAMS',
            v1_params + '_V2',
            v2_name.replace('_CTRL_CMD_', '_CTRL_') + '_PARAMS',
        ]
        seen = set()
        v2_params_try = []
        for p in v2_params_candidates:
            if p not in seen:
                seen.add(p)
                v2_params_try.append(p)

        # Find V2 params (also following aliases)
        v2_params = None
        v2_body = None
        v2_real_type = None
        for v2p in v2_params_try:
            resolved = resolve_struct(v2p)
            if resolved is not None:
                real_type, body, _ = resolved
                if 'NvP64' not in body:
                    v2_params = v2p
                    v2_body = body
                    v2_real_type = real_type
                    break

        if v2_params is None:
            print(f"  WARN: V1/V2 cmd pair found but no matching V2 params struct: "
                  f"{cmd_name} → tried {v2_params_try}", file=sys.stderr)
            continue

        candidates.append(V1V2Candidate(
            v1_cmd_name=cmd_name,
            v2_cmd_name=v2_name,
            v1_cmd_value=cmd_value,
            v2_cmd_value=v2_value,
            v1_params_type=v1_params,
            v2_params_type=v2_params,
            header_path=cmd_path,
        ))

    print(f"  Found {len(candidates)} V1→V2 rewrite candidates")
    return candidates, all_structs, all_aliases


def classify_v1v2_style(v1_body: str, v2_body: str) -> tuple[bool, str]:
    """
    Determine if a V1/V2 pair is info_style or caps_style.

    caps_style: V1 has capsTblSize (byte count), V2 has NvU8 capsTbl[N]
    info_style: V1 has listSize (item count), V2 has SomeStruct list[N]

    Returns (is_info_style, hint_string)
    """
    # Check for common info-style indicators
    if re.search(r'\blistSize\b|\bInfoListSize\b|\binfoListSize\b', v1_body, re.IGNORECASE):
        return True, "list_size_field"

    # Check for caps-style indicators
    if re.search(r'\bcapsTblSize\b|\bcapsTableSize\b', v1_body, re.IGNORECASE):
        return False, "caps_tbl_size_field"

    # Heuristic: if V2 has NvU8 array, it's caps; if it has a struct array, it's info
    if re.search(r'NvU8\s+\w+\s*\[', v2_body):
        return False, "v2_has_NvU8_array"

    # Default to info style if unclear
    return True, "default_assumed_info"


def analyze_v1_struct_fields(body: str) -> list[tuple[str, str]]:
    """
    Extract field declarations from a struct body.
    Returns list of (type, name) tuples in declaration order.

    Handles NV_DECLARE_ALIGNED(NvP64 name, 8) specially.
    """
    fields = []

    # Match NV_DECLARE_ALIGNED(Type name, alignment)
    aligned_re = re.compile(r'NV_DECLARE_ALIGNED\s*\(\s*(\w+)\s+(\w+)\s*,\s*\d+\s*\)')
    # Match simple "Type name;" declarations
    simple_re = re.compile(r'^\s*(Nv\w+|NvU\d+|NvS\d+|NvBool|NvHandle)\s+(\w+)\s*;', re.MULTILINE)

    # Process line by line to maintain order
    for line in body.split('\n'):
        line = line.strip()
        if not line or line.startswith('//') or line.startswith('/*'):
            continue

        m = aligned_re.search(line)
        if m:
            fields.append((m.group(1), m.group(2)))
            continue

        m = simple_re.match(line)
        if m:
            fields.append((m.group(1), m.group(2)))
            continue

    return fields


def resolve_v1v2_details_via_compile(
        src_root: Path,
        candidates: list[V1V2Candidate],
        all_structs: dict[str, tuple[str, Path]],
        all_aliases: dict[str, str] = None,
) -> list[V1V2Candidate]:
    """
    Use a C probe to get exact sizeof/offsetof for V1/V2 struct fields.
    """
    if not candidates:
        return candidates

    if all_aliases is None:
        all_aliases = {}

    def get_struct_body(type_name: str) -> str | None:
        """Get struct body, following aliases."""
        visited = set()
        current = type_name
        while current not in all_structs:
            if current in all_aliases and current not in visited:
                visited.add(current)
                current = all_aliases[current]
            else:
                return None
        body, _ = all_structs[current]
        return body

    include_dirs = [
        src_root / "src" / "common" / "sdk" / "nvidia" / "inc",
        src_root / "src" / "common" / "sdk" / "nvidia" / "inc" / "class",
        src_root / "src" / "common" / "sdk" / "nvidia" / "inc" / "ctrl",
        src_root / "kernel-open" / "common" / "inc",
        src_root / "src" / "common" / "inc",
        src_root / "src" / "nvidia" / "inc",
        src_root / "src" / "nvidia" / "arch" / "nvalloc" / "unix" / "include",
    ]

    # For each candidate, we need to find the NvP64 field name in V1
    v1_ptr_fields: dict[str, str] = {}  # v1_params_type → NvP64 field name
    v1_prefix_fields: dict[str, list[str]] = {}  # v1_params_type → fields before NvP64

    for cand in candidates:
        if cand.v1_params_type in all_structs:
            body = get_struct_body(cand.v1_params_type)
            if body:
                fields = analyze_v1_struct_fields(body)
                prefix = []
                ptr_name = None
                for ftype, fname in fields:
                    if ftype == 'NvP64':
                        ptr_name = fname
                        break
                    prefix.append(fname)
                if ptr_name:
                    v1_ptr_fields[cand.v1_params_type] = ptr_name
                    v1_prefix_fields[cand.v1_params_type] = prefix

    # Also detect info_style and find V2 data field
    v2_data_fields: dict[str, str] = {}  # v2_params_type → array field name
    for cand in candidates:
        if cand.v1_params_type in all_structs and get_struct_body(cand.v2_params_type):
            v1_body = get_struct_body(cand.v1_params_type)
            v2_body = get_struct_body(cand.v2_params_type)
            if v1_body and v2_body:
                cand.info_style, _ = classify_v1v2_style(v1_body, v2_body)
                ptr_name = v1_ptr_fields.get(cand.v1_params_type)
                if ptr_name:
                    array_re = re.compile(
                        rf'\b(\w+)\s+{re.escape(ptr_name)}\s*\[',
                    )
                    m = array_re.search(v2_body)
                    if m:
                        v2_data_fields[cand.v2_params_type] = ptr_name

    # Collect unique header includes needed
    needed_headers: set[Path] = set()
    for cand in candidates:
        needed_headers.add(cand.header_path)

    # Build C probe
    lines = [
        '#include <stdio.h>',
        '#include <stddef.h>',
        '',
        '#include "nvtypes.h"',
        '#include "nvos.h"',
        '#include "nv_escape.h"',
        '',
    ]

    included = set()
    for header in sorted(needed_headers):
        for inc_dir in include_dirs:
            try:
                rel = header.relative_to(inc_dir)
                inc_str = str(rel)
                if inc_str not in included:
                    lines.append(f'#include "{inc_str}"')
                    included.add(inc_str)
                break
            except ValueError:
                continue

    lines.extend(['', 'int main(void) {'])

    for cand in candidates:
        tag = f"{cand.v1_cmd_name}"
        ptr_field = v1_ptr_fields.get(cand.v1_params_type)
        v2_data_field = v2_data_fields.get(cand.v2_params_type)
        prefix_fields = v1_prefix_fields.get(cand.v1_params_type, [])

        if not ptr_field:
            lines.append(f'    /* SKIP {tag}: no NvP64 field found */')
            continue

        # Emit sizeof and offsetof for V1
        lines.append(f'    /* {tag} */')
        lines.append(f'    printf("{tag} v1_size %zu\\n", sizeof({cand.v1_params_type}));')
        lines.append(f'    printf("{tag} v2_size %zu\\n", sizeof({cand.v2_params_type}));')
        lines.append(f'    printf("{tag} v1_ptr_offset %zu\\n", '
                     f'offsetof({cand.v1_params_type}, {ptr_field}));')

        # Prefix size = offset of the NvP64 field (bytes before it)
        lines.append(f'    printf("{tag} v1_prefix_size %zu\\n", '
                     f'offsetof({cand.v1_params_type}, {ptr_field}));')

        # V2 data offset and size
        if v2_data_field:
            lines.append(f'    printf("{tag} v2_data_offset %zu\\n", '
                         f'offsetof({cand.v2_params_type}, {v2_data_field}));')
            lines.append(f'    printf("{tag} v2_data_size %zu\\n", '
                         f'sizeof((({cand.v2_params_type}*)0)->{v2_data_field}));')
        else:
            # Fallback: data starts after prefix fields, size = v2_size - prefix
            lines.append(f'    printf("{tag} v2_data_offset 0\\n");')
            lines.append(f'    printf("{tag} v2_data_size %zu\\n", '
                         f'sizeof({cand.v2_params_type}));')

        lines.append(f'    printf("{tag} info_style %d\\n", {1 if cand.info_style else 0});')
        lines.append('')

    lines.extend(['    return 0;', '}'])

    probe_src = '\n'.join(lines) + '\n'

    with tempfile.TemporaryDirectory(prefix="nvgpu_gen_v1v2_") as tmpdir:
        src_file = os.path.join(tmpdir, "probe_v1v2.c")
        bin_file = os.path.join(tmpdir, "probe_v1v2")

        with open(src_file, 'w') as f:
            f.write(probe_src)

        debug_copy = Path("nvgpu_gen_probe_v1v2.c")
        debug_copy.write_text(probe_src)
        print(f"  Wrote V1V2 probe source to {debug_copy} for debugging")

        inc_flags = []
        for d in include_dirs:
            if d.exists():
                inc_flags.extend(["-I", str(d)])

        cmd = ["gcc", "-o", bin_file, src_file] + inc_flags + [
            "-Wno-all",
            "-fsigned-char",
            "-D__packed=__attribute__((packed))",
            "-DNVRM",
            "-DNV_KERNEL_INTERFACE_LAYER",
        ]

        print(f"  Compiling V1V2 probe...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: V1V2 probe compilation failed:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return candidates

        result = subprocess.run([bin_file], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: V1V2 probe execution failed", file=sys.stderr)
            return candidates

    # Parse output
    probe_data: dict[str, dict[str, int]] = {}
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        parts = line.split()
        tag = parts[0]
        key = parts[1]
        value = int(parts[2])
        probe_data.setdefault(tag, {})[key] = value

    # Fill in candidate details
    for cand in candidates:
        tag = cand.v1_cmd_name
        if tag not in probe_data:
            continue
        d = probe_data[tag]
        cand.v1_size = d.get('v1_size', 0)
        cand.v2_size = d.get('v2_size', 0)
        cand.v1_userptr_offset = d.get('v1_ptr_offset', 0)
        cand.v1_prefix_size = d.get('v1_prefix_size', 0)
        cand.v2_data_offset = d.get('v2_data_offset', 0)
        cand.v2_data_size = d.get('v2_data_size', 0)
        cand.info_style = bool(d.get('info_style', 0))

    return candidates


# ---------------------------------------------------------------------------
# Emit nvgpu_v1v2_rewrites.h
# ---------------------------------------------------------------------------

def emit_v1v2_header(
        candidates: list[V1V2Candidate],
        out_dir: Path,
        driver_version: str,
):
    lines = [
        '/* SPDX-License-Identifier: GPL-2.0 */',
        '/*',
        f' * nvgpu_v1v2_rewrites.h — auto-generated from NVIDIA driver {driver_version}',
        ' *',
        ' * Generated by nvgpu_gen.py — DO NOT EDIT MANUALLY',
        ' *',
        ' * V1→V2 ioctl rewrite table for RM_CONTROL.',
        ' * Many RM_CONTROL commands have two variants:',
        ' *   V1: nested params contain a userspace pointer to a data buffer',
        ' *   V2: nested params contain inline data (no second-level pointer)',
        ' *',
        ' * The guest driver cannot forward V1 to the VMM because the VMM cannot',
        ' * dereference guest userspace pointers inside nested params.',
        ' */',
        '',
        '#ifndef NVGPU_V1V2_REWRITES_H',
        '#define NVGPU_V1V2_REWRITES_H',
        '',
        '#include <linux/types.h>',
        '',
        'struct nvgpu_v1v2_entry {',
        '    u32  v1_cmd;            /* RM_CONTROL cmd that carries V1 layout      */',
        '    u32  v2_cmd;            /* replacement cmd with inline V2 layout      */',
        '    u32  v2_size;           /* sizeof the V2 nested params struct         */',
        '    u32  v1_userptr_offset; /* byte offset of NvP64 in V1 nested params  */',
        '    u32  v1_copy_prefix;    /* bytes to copy from V1 start → V2 start    */',
        '    u32  v2_data_offset;    /* offset where result data begins in V2     */',
        '    u32  v2_data_size;      /* max bytes of result data to copy back     */',
        '    bool info_style;        /* true = listSize is item count (×8 bytes)  */',
        '};',
        '',
        'static const struct nvgpu_v1v2_entry nvgpu_v1v2_table[] = {',
        '',
    ]

    # Separate caps-style and info-style for readability
    caps_entries = [c for c in candidates if not c.info_style and c.v2_size > 0]
    info_entries = [c for c in candidates if c.info_style and c.v2_size > 0]
    skipped = [c for c in candidates if c.v2_size == 0]

    if caps_entries:
        lines.append('    /* ── GET_CAPS style: V1 = {capsTblSize, pad, NvP64 capsTbl} ── */')
        lines.append('    /*' + ' ' * 14 + 'v1_cmd      v2_cmd    v2sz  ptr  pfx  d_off d_sz info */')
        for cand in sorted(caps_entries, key=lambda c: c.v1_cmd_value):
            # Extract short name from command
            short_name = cand.v1_cmd_name.split('_CTRL_CMD_')[
                -1] if '_CTRL_CMD_' in cand.v1_cmd_name else cand.v1_cmd_name
            lines.append(
                f'    /* {short_name} */'
            )
            lines.append(
                f'    {{0x{cand.v1_cmd_value:08x}, 0x{cand.v2_cmd_value:08x}, '
                f'{cand.v2_size:>4}, {cand.v1_userptr_offset:>2}, '
                f'{cand.v1_prefix_size:>2}, {cand.v2_data_offset:>4}, '
                f'{cand.v2_data_size:>4}, false}},'
            )
        lines.append('')

    if info_entries:
        lines.append('    /* ── GET_INFO style: V1 = {listSize, pad, NvP64 list} ── */')
        for cand in sorted(info_entries, key=lambda c: c.v1_cmd_value):
            short_name = cand.v1_cmd_name.split('_CTRL_CMD_')[
                -1] if '_CTRL_CMD_' in cand.v1_cmd_name else cand.v1_cmd_name
            lines.append(
                f'    /* {short_name} */'
            )
            lines.append(
                f'    {{0x{cand.v1_cmd_value:08x}, 0x{cand.v2_cmd_value:08x}, '
                f'{cand.v2_size:>4}, {cand.v1_userptr_offset:>2}, '
                f'{cand.v1_prefix_size:>2}, {cand.v2_data_offset:>4}, '
                f'{cand.v2_data_size:>4}, true}},'
            )
        lines.append('')

    lines.extend([
        '};',
        '',
        '#define NVGPU_V1V2_TABLE_SIZE ARRAY_SIZE(nvgpu_v1v2_table)',
        '',
        'static inline const struct nvgpu_v1v2_entry *nvgpu_find_v1v2_rewrite(u32 cmd)',
        '{',
        '    int i;',
        '    for (i = 0; i < (int)NVGPU_V1V2_TABLE_SIZE; i++) {',
        '        if (nvgpu_v1v2_table[i].v1_cmd == cmd)',
        '            return &nvgpu_v1v2_table[i];',
        '    }',
        '    return NULL;',
        '}',
        '',
        '#endif /* NVGPU_V1V2_REWRITES_H */',
        '',
    ])

    out_path = out_dir / "nvgpu_v1v2_rewrites.h"
    out_path.write_text('\n'.join(lines))
    print(f"Wrote {out_path} ({len(caps_entries)} caps-style, {len(info_entries)} info-style, "
          f"{len(skipped)} skipped)")

    if skipped:
        print("  Skipped (no probe data):")
        for c in skipped:
            print(f"    {c.v1_cmd_name}")


# ---------------------------------------------------------------------------
# Phase 3: Multi-pointer RM_CONTROL detection
# ---------------------------------------------------------------------------

@dataclass
class MultiPointerCmd:
    """An RM_CONTROL command whose params have multiple NvP64 fields."""
    cmd_name: str
    cmd_value: int
    params_type: str
    nvp64_count: int
    nvp64_fields: list[str]
    header_path: Path
    has_v2: bool = False  # True if a _V2 variant exists
    v2_cmd_name: str = ""


def find_multi_pointer_commands(
        src_root: Path,
        all_cmds: dict[str, tuple[int, Path]],
        all_structs: dict[str, tuple[str, Path]],
        all_aliases: dict[str, str],
) -> list[MultiPointerCmd]:
    """
    Find RM_CONTROL commands whose nested params contain 2+ NvP64 fields.
    These cannot use the standard V1→V2 rewrite path.
    """

    def resolve_body(type_name: str) -> str | None:
        visited = set()
        current = type_name
        while current not in all_structs:
            if current in all_aliases and current not in visited:
                visited.add(current)
                current = all_aliases[current]
            else:
                return None
        body, _ = all_structs[current]
        return body

    def count_nvp64(body: str) -> tuple[int, list[str]]:
        """Count NvP64 fields and return their names."""
        fields = []
        for m in re.finditer(r'NvP64\s+(\w+)', body):
            fields.append(m.group(1))
        return len(fields), fields

    results = []

    for cmd_name, (cmd_value, cmd_path) in sorted(all_cmds.items()):
        # Skip V2 commands themselves
        if cmd_name.endswith('_V2') or cmd_name.endswith('_V3'):
            continue

        # Derive params type
        params_type = cmd_name.replace('_CTRL_CMD_', '_CTRL_') + '_PARAMS'
        body = resolve_body(params_type)
        if body is None:
            continue

        count, fields = count_nvp64(body)
        if count < 2:
            continue

        # Check if V2 exists
        v2_name = cmd_name + '_V2'
        has_v2 = v2_name in all_cmds

        results.append(MultiPointerCmd(
            cmd_name=cmd_name,
            cmd_value=cmd_value,
            params_type=params_type,
            nvp64_count=count,
            nvp64_fields=fields,
            header_path=cmd_path,
            has_v2=has_v2,
            v2_cmd_name=v2_name if has_v2 else "",
        ))

    print(f"  Found {len(results)} commands with 2+ NvP64 fields")
    return results


def emit_multi_pointer_report(
        commands: list[MultiPointerCmd],
        out_dir: Path,
        driver_version: str,
):
    """Emit an advisory report of multi-pointer commands."""
    lines = [
        f'# Multi-pointer RM_CONTROL commands — NVIDIA driver {driver_version}',
        f'# Generated by nvgpu_gen.py',
        f'#',
        f'# These commands have 2+ NvP64 fields in their params struct.',
        f'# They CANNOT use the standard V1→V2 rewrite path.',
        f'# Each needs either:',
        f'#   - A local intercept (synthesize response in guest driver)',
        f'#   - A working V2 variant (if one exists and is externally callable)',
        f'#   - Special multi-pointer marshalling',
        f'#',
        f'# Commands marked [HAS_V2] have a _V2 command variant that may work.',
        f'#',
        '',
    ]

    for cmd in sorted(commands, key=lambda c: c.cmd_value):
        v2_tag = f" [HAS_V2: {cmd.v2_cmd_name}]" if cmd.has_v2 else ""
        lines.append(f'0x{cmd.cmd_value:08x}  {cmd.cmd_name}{v2_tag}')
        lines.append(f'  params: {cmd.params_type}')
        lines.append(f'  NvP64 fields ({cmd.nvp64_count}): {", ".join(cmd.nvp64_fields)}')
        lines.append(f'  header: {cmd.header_path.name}')
        lines.append('')

    out_path = out_dir / "multi_pointer_commands.txt"
    out_path.write_text('\n'.join(lines))
    print(f"Wrote {out_path} ({len(commands)} commands)")


# ---------------------------------------------------------------------------
# Filter V1V2 candidates against multi-pointer list
# ---------------------------------------------------------------------------

def filter_v1v2_candidates(
        candidates: list[V1V2Candidate],
        multi_ptr_cmds: list[MultiPointerCmd],
        all_structs: dict[str, tuple[str, Path]],
        all_aliases: dict[str, str],
) -> tuple[list[V1V2Candidate], list[V1V2Candidate]]:
    """
    Remove V1→V2 candidates whose V1 params have multiple NvP64 fields.

    Returns (valid_candidates, rejected_candidates)
    """

    def resolve_body(type_name: str) -> str | None:
        visited = set()
        current = type_name
        while current not in all_structs:
            if current in all_aliases and current not in visited:
                visited.add(current)
                current = all_aliases[current]
            else:
                return None
        body, _ = all_structs[current]
        return body

    multi_ptr_values = {cmd.cmd_value for cmd in multi_ptr_cmds}

    valid = []
    rejected = []

    for cand in candidates:
        # Check if V1 params has multiple NvP64
        v1_body = resolve_body(cand.v1_params_type)
        if v1_body:
            count = len(re.findall(r'NvP64\s+\w+', v1_body))
            if count >= 2:
                print(f"  Filtering out {cand.v1_cmd_name}: V1 params has {count} NvP64 fields")
                rejected.append(cand)
                continue

        valid.append(cand)

    return valid, rejected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def detect_driver_version(src_root: Path) -> str:
    """Try to detect driver version from the source tree."""
    # Check version.mk or nv-kernel.o_binary version string
    version_mk = src_root / "version.mk"
    if version_mk.exists():
        content = version_mk.read_text()
        m = re.search(r'NVIDIA_VERSION\s*[?:]*=\s*(\S+)', content)
        if m:
            return m.group(1)

    # Fallback: directory name
    return src_root.name


def main():
    parser = argparse.ArgumentParser(
        description="Generate virtio-gpu-nv driver tables from NVIDIA source"
    )
    parser.add_argument(
        "--src", required=True, type=Path,
        help="Path to open-gpu-kernel-modules source root"
    )
    parser.add_argument(
        "--out", required=True, type=Path,
        help="Output directory for generated headers"
    )
    parser.add_argument(
        "--version", type=str, default=None,
        help="Override driver version string"
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["all", "rmalloc", "v1v2", "multiptr"],
        help="Which phase to run (default: all)"
    )

    args = parser.parse_args()
    src_root = args.src.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    driver_version = args.version or detect_driver_version(src_root)
    print(f"NVIDIA driver version: {driver_version}")
    print(f"Source root: {src_root}")
    print()

    # --- Phase 1: rmalloc classes ---
    if args.phase in ("all", "rmalloc"):
        print("=== Phase 1: RM_ALLOC class param sizes ===")
        entries = parse_resource_list(src_root)
        class_names = {e.external_class for e in entries}
        print(f"Resolving {len(class_names)} class names to numeric values...")
        hclass_map = find_hclass_defines(src_root, class_names)
        print(f"  Resolved {len(hclass_map)}/{len(class_names)} class names")

        print("Resolving param struct sizes via C probe compilation...")
        sizes = resolve_param_sizes_via_compile(src_root, entries, hclass_map)

        if sizes:
            emit_rmalloc_header(sizes, out_dir, driver_version)
        else:
            print("ERROR: No sizes resolved, cannot generate header", file=sys.stderr)
            sys.exit(1)
        print()

    # --- Phase 2 & 3: V1→V2 rewrites + multi-pointer detection ---
    if args.phase in ("all", "v1v2", "multiptr"):
        print("=== Phase 2/3: V1→V2 rewrites + multi-pointer detection ===")
        candidates, all_structs, all_aliases = find_v1v2_pairs(src_root)

        # We need all_cmds for Phase 3 — re-collect
        # (or better: have find_v1v2_pairs return it too)
        ctrl_dir = src_root / "src" / "common" / "sdk" / "nvidia" / "inc" / "ctrl"
        all_cmds = {}
        cmd_define_re = re.compile(
            r'^\s*#\s*define\s+(NV\w+_CTRL_CMD_\w+)\s+\(?(0x[0-9a-fA-F]+)U?\)?\s*',
            re.MULTILINE
        )
        for hfile in ctrl_dir.rglob("*.h"):
            try:
                content = hfile.read_text(errors='replace')
            except Exception:
                continue
            for m in cmd_define_re.finditer(content):
                all_cmds[m.group(1)] = (int(m.group(2), 16), hfile)

        # Phase 3: Find multi-pointer commands
        if args.phase in ("all", "multiptr"):
            print("\n--- Multi-pointer detection ---")
            multi_ptr = find_multi_pointer_commands(
                src_root, all_cmds, all_structs, all_aliases
            )
            emit_multi_pointer_report(multi_ptr, out_dir, driver_version)

        # Filter V1V2 candidates against multi-pointer list
        if args.phase in ("all", "v1v2") and candidates:
            if args.phase == "all":
                valid, rejected = filter_v1v2_candidates(
                    candidates, multi_ptr, all_structs, all_aliases
                )
                if rejected:
                    print(f"\n  Removed {len(rejected)} multi-pointer entries from V1V2 table")
                candidates = valid

            candidates = resolve_v1v2_details_via_compile(
                src_root, candidates, all_structs, all_aliases
            )
            emit_v1v2_header(candidates, out_dir, driver_version)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
