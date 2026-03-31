# virtio-gpu-nv

**A virtio device for near-native NVIDIA GPU access in KVM virtual machines.**

virtio-gpu-nv forwards NVIDIA kernel driver ioctls between a Linux guest
and the host at the **driver ABI level**, bypassing API-level translation
entirely. The guest runs NVIDIA's own user-mode drivers unmodified.

The primary target is **headless cloud gaming**: a compositor inside the
VM renders, composites, and encodes frames via the GPU, then streams
compressed video to a remote display. The VM has no physical monitor.

> **This is a design document and proposal.** Nothing here is implemented
> yet. The design is speculative and subject to change at any time as
> understanding deepens and prototyping reveals new constraints.

---

## Why

### The Streaming Pipeline We Want

```text
Guest VM (headless, no physical display)
──────────────────────────────────────────

  Game / application
    │ Vulkan or OpenGL
    ▼
  Wayland compositor (guest-side)
    │ composites all windows
    │ CUDA zero-copy import of composed frame
    ▼
  NVENC hardware encoder (guest-side)
    │ H.264 / H.265 bitstream (~100 KB per frame)
    ▼
  Stream to remote client
```

The entire render → composite → encode pipeline runs **on the GPU,
inside the guest**. Only the compressed bitstream leaves. This requires
the guest to have **real, driver-level access** to GPU resources: buffer
handles, fences, CUDA device pointers, NVENC sessions.

### Why Existing Solutions Fall Short

**virtio-gpu + Venus (API-level translation)**

Venus works by serializing every Vulkan or OpenGL API call in the guest,
transporting it over virtio, deserializing it on the host, and replaying
it against the host driver. This has three problems for our use case:

1. **Latency compounds on draw-call-heavy workloads.**
   Games routinely issue 1,000–5,000 draw calls per frame, plus
   pipeline binds, descriptor updates, and render pass transitions.
   Each one is serialized and replayed individually. At 60 fps with a
   16.6 ms frame budget, even 1–3 ms of serialization overhead per
   frame is 6–18% of the budget gone before any GPU work happens.

2. **CPU overhead is significant.**
   Serialization, transport, and replay all burn host CPU cycles that
   the game itself needs. In a cloud gaming context where compute
   resources are billed and finite, this waste matters. Venus can
   reduce draw-call-heavy workloads to 65–85% of bare-metal
   performance, with the gap being almost entirely CPU-side overhead.

3. **Guest-side encoding is not viable.**
   With Venus, GPU buffers are owned by the **host**. The guest
   compositor cannot see or import them. OpenGL buffer tracking is
   particularly broken: GL's implicit synchronization model does not
   survive API-level serialization and host-side replay. There is no
   practical path to get a `CUdeviceptr` in the guest that points to a
   Venus-managed buffer, which means the guest cannot feed frames to
   NVENC without a full CPU-side readback and copy.

**DRM native context (Intel / AMD)**

For Intel and AMD, virtio-gpu supports "native context" mode: the guest
runs the real Mesa driver, builds GPU command buffers locally, and only
queue submissions cross the VM boundary. This gives 95–99% of bare-metal
performance. Guest-side buffer ownership and encoding work correctly.

This does not exist for NVIDIA on Linux.

**VFIO passthrough**

Full GPU passthrough gives native performance and a complete driver stack
in the guest, but dedicates the entire GPU to one VM. In multi-tenant or
cloud environments, this is often not an option.

### What virtio-gpu-nv Does Differently

Instead of translating at the **graphics API** level (Vulkan/GL calls),
virtio-gpu-nv translates at the **kernel driver** level (ioctls to
`/dev/nvidia*`). The guest runs NVIDIA's real user-mode libraries, which
build GPU command buffers **locally in the guest** — no serialization of
individual draw calls:

```text
                    Venus                  virtio-gpu-nv
                    ──────────────         ──────────────────────

Per draw call:      serialize +            local function call
                    transport +            (~10 ns, no VM exit)
                    deserialize +
                    replay

Per frame           ~2,000 messages        ~5–20 messages
  boundary          (one per API call)     (queue submits + allocs)
  crossings

GPU command         generated on HOST      generated in GUEST
  buffers           after replay           by NVIDIA's own compiler

CPU overhead        serialization +        near zero for rendering
                    deserialization        (only ioctl forwarding)

Guest buffer        HOST owns buffers      GUEST owns buffers
  ownership         compositor can't       compositor has full
                    track them             visibility and control

Guest NVENC         not viable             works (real CUDA interop)
```

---

## How It Works

Three components:

### 1. Virtio Device (in the VMM)

A custom virtio device with:

- A **control virtqueue** for request/response messages (open, close,
  ioctl, mmap setup).
- A **shared memory (SHM) BAR** — a contiguous memory region accessible
  to both guest and host, used for GPU buffer mappings.

### 2. Guest Kernel Driver (`virtio_gpu_nv.ko`)

A Linux kernel module that registers:

- `/dev/nvidiactl` — driver control
- `/dev/nvidia0`, `/dev/nvidia1`, … — per-GPU devices
- `/dev/nvidia-uvm` — CUDA memory management

When an application calls `ioctl()` or `mmap()` on these devices, the
driver serializes the request and sends it over the virtqueue. For
`mmap()`, it maps the appropriate SHM BAR range into the process's
address space with correct caching attributes.

The guest driver is **not ABI-aware**. It copies raw bytes. All
ABI-specific logic lives in the backend.

### 3. VMM Backend (Rust, in libkrun)

Receives requests from the virtqueue and:

- Maps guest handles to real host `/dev/nvidia*` file descriptors.
- Performs ABI-aware translation of ioctl parameters (pointer and FD
  translation), ported from gVisor nvproxy's logic.
- Calls `ioctl(2)` on the host device FDs.
- For GPU memory mappings, `mmap()`s the host device FD into the SHM
  region so the guest can access it directly.

```text
┌─ Guest ─────────────────────────────────────────────────┐
│                                                         │
│  Game → NVIDIA Vulkan/GL/CUDA → ioctl(/dev/nvidia*)     │
│                                       │                 │
│  virtio_gpu_nv.ko                     │                 │
│    serialize → virtqueue              │                 │
│    mmap → SHM BAR (remap_pfn_range)   │                 │
└───────────────────────────────────────┼─────────────────┘
                                        │ VM exit
┌───────────────────────────────────────▼─────────────────┐
│  libkrun VMM                                            │
│                                                         │
│  virtio-gpu-nv backend                                  │
│    ├─ translate handles → host FDs                      │
│    ├─ translate embedded FDs/pointers in ioctl params   │
│    ├─ ioctl(host /dev/nvidia*, ...)                     │
│    └─ mmap(host FD) → SHM region (guest-accessible)    │
│                                                         │
│  Host NVIDIA driver → GPU hardware                      │
└─────────────────────────────────────────────────────────┘
```

---

## What Works, What Doesn't

### Supported (Target)

- Vulkan rendering (all extensions that work without `/dev/nvidia-drm`)
- OpenGL rendering (headless EGL, GLX fallback via X11 proxy)
- CUDA device memory allocations
- CUDA ↔ Vulkan/GL interop (zero-copy, GPU-side pointers)
- NVENC hardware encoding from CUDA device pointers
- NVDEC hardware decoding

### Not Supported

- `cudaMallocManaged()` / full unified virtual memory
  (known flaky even in gVisor KVM; not needed for NVENC pipeline)
- `/dev/nvidia-drm` and `/dev/nvidia-modeset`
  (physical display output; not needed for headless streaming)
- Multi-GPU, MIG, SR-IOV
- Arbitrary NVIDIA driver versions
  (each version must be explicitly supported, same as nvproxy)

---

## Performance Expectations

These are design targets, not measurements. No code exists yet.

|                             | Venus                   | virtio-gpu-nv (target) | Bare metal |
| --------------------------- | ----------------------- | ---------------------- | ---------- |
| GPU-bound (heavy shaders)   | 90–97%                  | 97–100%                | 100%       |
| CPU-bound (many draw calls) | 65–85%                  | 95–99%                 | 100%       |
| Shader compilation stutter  | +10–50 ms per shader    | <1 ms overhead         | 0          |
| Frame latency overhead      | +1–3 ms                 | +0.05–0.1 ms           | 0          |
| CPU overhead for rendering  | High (serialize/replay) | Near zero              | Zero       |
| Guest-side NVENC            | Not viable              | Works (zero-copy)      | Works      |

The fundamental difference: Venus crosses the VM boundary **per API
call** (thousands per frame). virtio-gpu-nv crosses it **per ioctl**
(tens per frame). Games do many small, consecutive GPU tasks where
per-call latency adds up. Eliminating that overhead is the point.

---

## NVIDIA ABI and Driver Versions

NVIDIA's kernel driver ABI is **not stable**. Ioctl struct layouts can
change between releases. virtio-gpu-nv handles this the same way gVisor
nvproxy does: each supported driver version has an explicit ABI
definition mapping ioctl numbers to handler functions and struct layouts.

When a new NVIDIA driver ships, adding support means:

1. Diff the open kernel modules repo for struct changes.
2. Update ABI definitions.
3. Test.

This is a maintenance burden, but it is bounded (a few hours per release)
and can directly follow nvproxy's upstream changes.

---

## Intended Use Case

A headless KVM virtual machine running on a host with an NVIDIA GPU:

- The VM runs a game or graphical application.
- A Wayland compositor inside the VM manages windows and composites
  frames.
- The compositor encodes frames with NVENC via CUDA zero-copy and
  streams the compressed bitstream to a remote client.
- The host has no monitor connected to the GPU.
- Compute resources (CPU, GPU, memory) are constrained and billed;
  overhead must be minimized.

This is the architecture used by cloud gaming platforms, remote desktop
solutions, and GPU-accelerated development environments that stream to
thin clients.

---

## Status

**Stage**: Design and proposal. No implementation exists.

The repository will eventually contain:

- The virtio device backend (Rust, targeting libkrun).
- The Linux guest kernel driver.
- ABI definitions for supported NVIDIA driver versions.

---

## Prior Art and Acknowledgements

This project builds on ideas and code from several existing systems:

### gVisor nvproxy

[gVisor](https://github.com/google/gvisor)'s `nvproxy` is the direct
inspiration for virtio-gpu-nv. nvproxy forwards NVIDIA ioctls from
sandboxed containers to the host driver, handling ABI versioning,
pointer/FD translation, and GPU mmap management. It supports Vulkan,
OpenGL, CUDA, and NVENC in production today.

virtio-gpu-nv adapts nvproxy's approach — ioctl categorization, ABI
version tables, handler dispatch — for a full VM context with virtio
transport. nvproxy's Go ABI definitions (in `pkg/abi/nvgpu`) and handler
logic (in `pkg/sentry/devices/nvproxy`) are the primary reference for
the backend implementation.

nvproxy also proves that Vulkan and NVENC work **without**
`/dev/nvidia-drm` or `/dev/nvidia-modeset` — only `/dev/nvidiactl`,
`/dev/nvidia#`, and `/dev/nvidia-uvm` are needed.

When gVisor runs on its KVM platform, the architecture is structurally
the same as what virtio-gpu-nv proposes: application code runs in KVM
guest mode, ioctls cause VM exits, and the sentry (a host process)
forwards them to host NVIDIA devices. The difference is that gVisor has
no guest kernel; virtio-gpu-nv adds a guest kernel module with virtio
transport.

### WSL2 /dev/dxg

Microsoft and NVIDIA built `/dev/dxg` (`dxgkrnl`) for WSL2: a guest
kernel module that forwards GPU operations to the Windows host over
Hyper-V VMBus. This is a production system (~30K lines of C) that proves
driver-level GPU proxying works across a real virtualization boundary. It
supports CUDA and DirectX compute, though not display.

virtio-gpu-nv targets a smaller scope (Linux-to-Linux, open ecosystem,
no DirectX) but shares the same core architecture.

### DRM Native Context (Intel / AMD)

virtio-gpu's native context mode for Intel and AMD achieves the same goal
as virtio-gpu-nv: the guest runs the real GPU driver and builds command
buffers locally; only submissions cross the VM boundary. This delivers
95–99% of bare-metal performance. virtio-gpu-nv aims to provide
equivalent capabilities for NVIDIA GPUs where no DRM native context
exists.

### libkrun

[libkrun](https://github.com/containers/libkrun) is the target VMM.
virtio-gpu-nv's backend will integrate as a virtio device alongside
libkrun's existing device implementations (virtio-gpu, virtio-net,
virtio-vsock, etc.).

---

## See Also

- [Architecture.md](Architecture.md) — detailed technical design:
  guest driver, backend, virtio protocol, memory model, ABI handling,
  CUDA/NVENC integration.
