# Project: virtio-gpu-nv

You are helping me build `virtio-gpu-nv`, a virtio device + Linux guest
kernel driver that forwards NVIDIA kernel driver ioctls between a KVM
guest and host, giving the guest near-native GPU performance.

## What this is

A custom virtio device (backend in Rust) and a Linux kernel module
(guest driver in C) that together let unmodified NVIDIA user-mode
libraries (`libvulkan_nvidia.so`, `libGL_nvidia.so`, `libcuda.so`,
NVENC) run inside a KVM guest by proxying their `/dev/nvidia*` ioctls
to the host.

Target use case: headless cloud gaming. A Wayland compositor inside the
VM renders, composites, and encodes frames via NVENC (CUDA zero-copy),
then streams the compressed bitstream to a remote client. No physical
monitor.

## Why not existing solutions

- **virtio-gpu + Venus**: Serializes every Vulkan/GL API call across the
  VM boundary (thousands per frame). High CPU overhead, 1-3ms latency
  per frame. Buffers owned by host — guest compositor can't track them.
  OpenGL buffer tracking is broken. Guest-side NVENC not viable because
  guest can't get CUdeviceptr for host-owned buffers.

- **DRM native context**: Exists for Intel/AMD (guest runs real driver,
  builds command buffers locally, 95-99% bare metal). Does not exist for
  NVIDIA on Linux.

- **VFIO passthrough**: Full native but dedicates entire GPU to one VM.

## Why this works

gVisor's `nvproxy` already does exactly this for containers: intercepts
NVIDIA ioctls from sandboxed apps and forwards them to host
`/dev/nvidia*`. On gVisor's KVM platform, app code runs in KVM guest
mode, ioctls cause VM exits, sentry forwards them. This is structurally
identical to a VM. The only difference: gVisor has no guest kernel. We
add a guest kernel module with virtio transport.

Key facts from nvproxy that validate the design:

- Vulkan, OpenGL, CUDA, and NVENC all work with only `/dev/nvidiactl`,
  `/dev/nvidia#`, and `/dev/nvidia-uvm`. No `/dev/nvidia-drm` or
  `/dev/nvidia-modeset` needed.
- The NVENC zero-copy pipeline (Vulkan render → CUDA interop →
  NVENC encode) does NOT use `cudaMallocManaged()`. It uses device
  memory allocations and `cuGraphicsMapResources` which return GPU
  virtual addresses (CUdeviceptr), not CPU addresses. No UVM page
  faults involved.
- CUDA init maps ~2MB of `/dev/nvidia-uvm` at a fixed address. In
  gVisor KVM this can conflict with sentry address space. In a real VM
  it doesn't because the guest has its own page tables.

## Architecture

Three components:

### 1. Protocol (shared contract)

`#[repr(C)]` message types used by both guest driver and backend.
Must be defined first. Guest driver uses a matching C header.

Messages: OPEN, CLOSE, IOCTL (request/response), with mmap metadata
returned in ioctl responses when the backend sets up a GPU mapping.

### 2. Guest kernel driver (`virtio_gpu_nv.ko`, C)

- Binds to virtio device
- Registers char devices: `/dev/nvidiactl`, `/dev/nvidia0..N`,
  `/dev/nvidia-uvm`
- `nv_ioctl()`: copies raw bytes from userspace, sends over virtqueue,
  waits for response, copies back. **NOT ABI-aware** — just forwards
  bytes.
- `nv_mmap()`: after a mapping ioctl (like `NV_ESC_RM_MAP_MEMORY`),
  the backend returns SHM offset + length + memory type. Guest driver
  calls `remap_pfn_range()` to map the SHM BAR range into the
  process's VMA with correct `pgprot` (UC/WC/WB).
- `nv_poll()`: for async GPU events (fdnotifier equivalent)

### 3. VMM backend (Rust, integrates with libkrun)

- Holds real host FDs for `/dev/nvidia*`
- Handle table: maps guest handles → host RawFd
- ABI-aware ioctl dispatch (ported from gVisor nvproxy):
  - Simple ioctls: copy params, call host ioctl, copy back
  - FD-carrying ioctls: translate embedded guest handle → host fd
  - Mapping ioctls: call host ioctl, mmap host fd into SHM region,
    return SHM offset to guest
  - Nested dispatch: `NV_ESC_RM_CONTROL` dispatches by control cmd,
    `NV_ESC_RM_ALLOC` dispatches by allocation class
- SHM region allocator: manages offsets within the shared memory BAR
- Driver ABI versioning: `HashMap<DriverVersion, DriverAbi>`, each
  version maps ioctl numbers to handlers. Ported from nvproxy.

### Memory model

- SHM BAR: contiguous region in VMM address space, mapped into guest
  physical memory via KVM memslot
- When backend handles `NV_ESC_RM_MAP_MEMORY`:
  1. Calls `ioctl()` on host fd
  2. Allocates region in SHM BAR
  3. `mmap(MAP_SHARED|MAP_FIXED, host_fd, 0)` into SHM region
  4. Returns offset + length + memory type to guest
- Guest driver's `nv_mmap()` calls `remap_pfn_range()` to expose that
  SHM range to userspace with correct caching

## Project structure

```
virtio-gpu-nv/
├── README.md
├── ARCHITECTURE.md
├── Cargo.toml                    # workspace
│
├── crates/
│   ├── protocol/                 # shared message definitions
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       └── messages.rs       # #[repr(C)] request/response types
│   │
│   ├── abi/                      # NVIDIA ioctl struct definitions
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── types.rs          # NvHandle, NV_STATUS, common types
│   │       ├── ioctl.rs          # NV_ESC_* constants, ioctl number helpers
│   │       ├── version.rs        # DriverVersion, version detection
│   │       └── versions/
│   │           ├── mod.rs        # version selection
│   │           └── v535_129_03.rs # first supported version
│   │
│   └── device/                   # VMM backend
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           ├── virtio.rs         # virtio device trait, queue handling
│           ├── nvidia.rs         # ioctl dispatch, handle table
│           ├── shm.rs            # SHM region allocator
│           └── mmap.rs           # mmap context tracking
│
└── guest-driver/                 # Linux kernel module (C)
    ├── Kbuild
    ├── Makefile
    ├── virtio_gpu_nv.h           # shared types + protocol header
    ├── virtio_gpu_nv_main.c      # module init, virtio probe
    ├── virtio_gpu_nv_ioctl.c     # ioctl: copy → virtqueue → wait → copy back
    ├── virtio_gpu_nv_mmap.c      # mmap: remap_pfn_range from SHM BAR
    └── virtio_gpu_nv_vq.c        # virtqueue send/receive helpers
```

**IMPORTANT**: This is a standalone project. It is NOT a fork of libkrun.
It produces Rust crates that libkrun can depend on. Do not clone or copy
libkrun source into this repo. The integration point is that libkrun's
device tree will eventually import the `device` crate.

## Development phases (vertical slices)

Each phase is testable end-to-end. Do not build one component in
isolation.

### Phase 1: Protocol + open/close

- Define protocol messages in `crates/protocol/`
- Backend: accept OPEN request, open real `/dev/nvidiactl` on host,
  return handle
- Guest driver: register `/dev/nvidiactl`, on open send OPEN via
  virtqueue, receive handle
- Mirror protocol as C header for guest driver
- **Test**: guest can `open()` and `close()` `/dev/nvidiactl`

### Phase 2: Simple ioctl passthrough

- Add IOCTL message type to protocol
- Backend: for one known simple ioctl (e.g. `NV_ESC_CHECK_VERSION_STR`),
  forward raw bytes to host, return response
- Guest driver: copy ioctl bytes from userspace, send, wait, copy back
- Port minimal ABI types from nvproxy's `pkg/abi/nvgpu`
- **Test**: guest can query driver version via ioctl

### Phase 3: Core Vulkan ioctl set + SHM mmap

- Port ioctl handlers needed for Vulkan init:
  `NV_ESC_RM_ALLOC`, `NV_ESC_RM_CONTROL`, `NV_ESC_RM_MAP_MEMORY`,
  `NV_ESC_RM_FREE`, `NV_ESC_REGISTER_FD`, `NV_ESC_ALLOC_OS_EVENT`,
  `NV_ESC_FREE_OS_EVENT`, `NV_ESC_CARD_INFO`, etc.
- FD translation for FD-carrying ioctls
- SHM BAR region + mmap forwarding
- ABI definitions for one driver version (535.129.03 or similar)
- **Test**: `vulkaninfo` or `vkcube` runs in guest

### Phase 4: CUDA + NVENC

- Add `/dev/nvidia-uvm` support (open/ioctl/mmap)
- Port UVM init ioctls from nvproxy
- Port CUDA interop RM control commands
- **Test**: simple CUDA program runs
- **Test**: NVENC encodes a test frame from CUDA device pointer

### Phase 5: Compositor integration

- Full pipeline: app render → compositor → CUDA import → NVENC → stream
- Test with real compositor and application

## Key references

### gVisor nvproxy (primary source for porting)

- ABI definitions: `pkg/abi/nvgpu/` (Go structs for NVIDIA ioctl params)
- Frontend ioctl handlers: `pkg/sentry/devices/nvproxy/frontend.go`
- Frontend mmap: `pkg/sentry/devices/nvproxy/frontend_mmap.go`
- Frontend mmap unsafe: `pkg/sentry/devices/nvproxy/frontend_mmap_unsafe.go`
- UVM handlers: `pkg/sentry/devices/nvproxy/uvm.go`
- Version/ABI tables: `pkg/sentry/devices/nvproxy/version.go`
- Object tracking: `pkg/sentry/devices/nvproxy/object.go`
- Main struct: `pkg/sentry/devices/nvproxy/nvproxy.go`
- Handler types: `pkg/sentry/devices/nvproxy/handlers.go`

### libkrun (target VMM)

- Existing virtio-gpu device: `src/devices/src/virtio/gpu/`
- Virtio device trait and MMIO: `src/devices/src/virtio/`

### NVIDIA open kernel modules

- https://github.com/NVIDIA/open-gpu-kernel-modules
- Ioctl definitions: `kernel-open/common/inc/nv-ioctl-numbers.h`
- Frontend dispatch: `kernel-open/nvidia/nv.c:nvidia_ioctl()`

## Rules

1. This is a standalone project with its own repo and workspace.
   Never fork or embed libkrun.
2. Protocol definitions come first. Both sides depend on them.
3. Every phase must be testable end-to-end (guest + host together).
4. Guest driver is NOT ABI-aware. It forwards raw bytes. ABI logic
   is only in the backend.
5. Port from nvproxy, don't reinvent. Their ioctl categorization
   (simple, FD-carrying, pointer-carrying, nested dispatch) is correct.
6. Start with ONE driver version. Add more later.
7. Use `#[repr(C)]` for all protocol and ABI structs.
8. The SHM BAR is sized at VM creation time and is the only mechanism
   for sharing GPU mmap regions with the guest.

Let's start with Phase 1.
