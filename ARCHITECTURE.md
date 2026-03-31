# virtio-gpu-nv Architecture

This document describes the architecture of `virtio-gpu-nv`: a virtio device
and guest kernel driver that forward NVIDIA kernel driver ioctls between a KVM
guest and the host, giving the guest driver-level control over GPU resources.

Everything described here follows directly from the design discussions that
motivated this project. If something is speculative or unresolved, it is
marked as such.

---

## 1. System Model

Concrete assumptions:

- **Host**: Linux, running official NVIDIA proprietary driver with open kernel
  modules (`nvidia.ko`, `nvidia-uvm.ko`). The host exposes
  `/dev/nvidiactl`, `/dev/nvidia0`, and `/dev/nvidia-uvm`.
- **Guest**: Linux, with NVIDIA user-mode libraries installed
  (`libvulkan_nvidia.so`, `libGL_nvidia.so`, `libcuda.so`, NVENC libraries).
  A custom `virtio-gpu-nv` kernel module replaces the real NVIDIA kernel
  modules inside the guest.
- **Hypervisor**: KVM.
- **VMM**: libkrun (Rust). The virtio-gpu-nv backend runs inside the VMM
  process.

From the guest's perspective, it looks like a normal NVIDIA driver stack.
All real hardware access happens on the host.

```text
Guest                                    Host
──────────────────────────────           ──────────────────────────
App (Vulkan / GL / CUDA / NVENC)
  │
  │ ioctl(/dev/nvidia*)
  ▼
virtio-gpu-nv guest driver
  │ serialize request
  │ virtqueue
  ▼                                      virtio-gpu-nv backend
  ═══════ VM exit ═══════════════════►     │ deserialize request
                                           │ translate handles/FDs
                                           │ ioctl(host_fd, ...)
                                           ▼
                                         NVIDIA KMD → GPU hardware
```

---

## 2. Why This Architecture

This section recaps the reasoning. Skip to Section 3 if you just want the
design.

### The Venus problem

With virtio-gpu + Venus:

- Every Vulkan/GL API call is serialized in the guest, transported over
  virtio, deserialized on the host, and replayed against the host driver.
- GPU buffers are owned by the **host**. The guest compositor cannot
  reliably track buffer state, especially for OpenGL (implicit sync).
- Guest-side NVENC encoding is not viable because the guest never holds
  real GPU pointers to import into CUDA.

### The DRM native context comparison

Intel and AMD have "DRM native context" for virtio-gpu, where the guest
runs the real Mesa driver and builds GPU command buffers locally. Only
queue submissions cross the VM boundary. This gives 95-99% of bare-metal
performance and correct guest buffer ownership.

NVIDIA has no equivalent in the open ecosystem.

### What gVisor nvproxy proves

gVisor's `nvproxy` already forwards NVIDIA ioctls from sandboxed apps to
the host driver. On gVisor's KVM platform, app code runs in KVM guest
mode, ioctls cause VM exits, and the sentry (host process) forwards them
to `/dev/nvidia*` on the host. GPU mmap regions are exposed to the guest
via KVM memslots.

This is **structurally the same** as what a VM would do. The one
difference: gVisor has no guest kernel, so the sentry catches VM exits
directly. In a real VM, we need a guest kernel module to bridge userspace
to the VMM via virtio. That is a well-understood pattern (virtio-gpu,
virtio-net, virtio-blk all do it).

### Why not /dev/nvidia-drm or /dev/nvidia-modeset

gVisor exposes only `/dev/nvidiactl`, `/dev/nvidia-uvm`, and
`/dev/nvidia#`, and Vulkan works. NVIDIA's `libvulkan_nvidia.so`
initializes with just `/dev/nvidiactl` + `/dev/nvidia0`. When
`/dev/dri/*` is absent, display-output extensions are disabled but all
rendering, compute, and encoding extensions remain functional.

Since we target offscreen rendering + streaming (not driving a physical
monitor from the VM), DRM and modeset are not needed.

---

## 3. Components

### 3.1 Virtio Device

A custom virtio device exposed to the guest via MMIO transport (libkrun's
standard approach).

**Virtqueues:**

- **Control queue**: request/response pairs for all operations (open,
  close, ioctl, mmap setup/teardown).

**Shared memory region (SHM BAR):**

- A contiguous region in the VMM's virtual address space.
- Mapped into guest physical memory as a virtio SHM region.
- All GPU mmap regions are placed here by the backend.
- Both guest CPU and host CPU can access the same physical pages.
- Sized at VM creation time (e.g. 2-8 GiB depending on workload).

### 3.2 Guest Kernel Driver

A Linux kernel module (`virtio_gpu_nv.ko`) that:

1. Binds to the virtio-gpu-nv device.
2. Registers character devices that mimic the real NVIDIA devices:
   - `/dev/nvidiactl` (control)
   - `/dev/nvidia0`, `/dev/nvidia1`, ... (per-GPU)
   - `/dev/nvidia-uvm` (CUDA memory management)
3. Implements `open`, `release`, `unlocked_ioctl`, `mmap`, and `poll`.
4. Forwards operations to the backend via the control virtqueue.

The guest driver is intentionally **not ABI-aware**. It copies raw ioctl
parameter bytes from userspace and sends them to the backend. All
ABI-specific logic (struct layout, pointer/FD translation) lives in the
backend.

### 3.3 VMM Backend

A Rust component inside libkrun that:

1. Holds real host file descriptors for `/dev/nvidia*` devices.
2. Receives requests from the control virtqueue.
3. Performs ABI-aware ioctl translation (ported from gVisor nvproxy's
   logic).
4. Calls `ioctl(2)` and `mmap(2)` on host device FDs.
5. Manages the SHM BAR region for GPU memory mappings.

---

## 4. Guest Kernel Driver Design

### 4.1 Device Registration

On probe, the driver discovers the virtio device, maps the SHM BAR, and
registers character devices:

```rust
// Pseudocode for guest kernel module logic.
// Actual implementation may use C or Rust-for-Linux bindings.

struct VirtioNvDev {
    vdev: VirtioDevice,
    ctrl_vq: Virtqueue,
    shm_base: *mut u8,      // SHM BAR kernel mapping
    shm_phys: u64,          // SHM BAR guest-physical address
    shm_size: u64,
}

struct NvFile {
    vndev: Arc<VirtioNvDev>,
    handle: u32,             // backend-assigned handle for this open FD
    dev_kind: DevKind,       // Ctl, Gpu(index), Uvm

    // Set by the backend after a mapping ioctl (e.g. NV_ESC_RM_MAP_MEMORY):
    mmap_shm_offset: Option<u64>,
    mmap_length: Option<u64>,
    mmap_mem_type: MemType,  // UC, WC, WB
}

enum DevKind {
    Ctl,
    Gpu { index: u32 },
    Uvm,
}

enum MemType {
    Uncached,
    WriteCombine,
    WriteBack,
}
```

### 4.2 Open / Close

```rust
fn nv_open(dev_kind: DevKind) -> Result<NvFile> {
    let req = VirtioNvOpenReq {
        dev_kind,
    };
    let resp: VirtioNvOpenResp = virtqueue_send_and_wait(&req)?;

    Ok(NvFile {
        handle: resp.handle,
        dev_kind,
        mmap_shm_offset: None,
        mmap_length: None,
        mmap_mem_type: MemType::Uncached,
        // ...
    })
}

fn nv_release(nf: &NvFile) -> Result<()> {
    let req = VirtioNvCloseReq {
        handle: nf.handle,
    };
    virtqueue_send_and_wait(&req)?;
    Ok(())
}
```

### 4.3 Ioctl

The guest driver does not interpret ioctl parameters. It copies raw bytes
and forwards them:

```rust
fn nv_ioctl(nf: &NvFile, cmd: u32, user_arg: *mut u8) -> Result<i32> {
    let nr = ioc_nr(cmd);
    let size = ioc_size(cmd);

    let mut params = vec![0u8; size as usize];
    copy_from_user(&mut params, user_arg, size)?;

    let req = VirtioNvIoctlReq {
        handle: nf.handle,
        cmd,
        nr,
        param_size: size,
        params,
    };

    let resp: VirtioNvIoctlResp = virtqueue_send_and_wait(&req)?;

    if resp.param_size > 0 {
        copy_to_user(user_arg, &resp.params, resp.param_size)?;
    }

    // If the backend tells us this ioctl set up an mmap context,
    // store the metadata for the subsequent mmap() call.
    if let Some(mmap_info) = resp.mmap_info {
        nf.mmap_shm_offset = Some(mmap_info.shm_offset);
        nf.mmap_length = Some(mmap_info.length);
        nf.mmap_mem_type = mmap_info.mem_type;
    }

    Ok(resp.ret)
}
```

### 4.4 Mmap

This is the part that solves the memory problem we discussed. The backend
has already `mmap`'d the host device FD into the SHM region. The guest
driver just maps the corresponding SHM BAR range into the process's
address space with correct caching attributes:

```rust
fn nv_mmap(nf: &NvFile, vma: &mut VmAreaStruct) -> Result<()> {
    let shm_offset = nf.mmap_shm_offset
        .ok_or(Error::EINVAL)?;
    let length = nf.mmap_length
        .ok_or(Error::EINVAL)?;

    let requested_len = vma.end - vma.start;
    if requested_len != length as usize {
        return Err(Error::EINVAL);
    }

    // Set page protection based on memory type from the backend.
    // This directly solves the gVisor KVM memory-type issue:
    // guest page tables get correct caching attributes.
    match nf.mmap_mem_type {
        MemType::Uncached => {
            vma.page_prot = pgprot_noncached(vma.page_prot);
        }
        MemType::WriteCombine => {
            vma.page_prot = pgprot_writecombine(vma.page_prot);
        }
        MemType::WriteBack => {
            // default, no change needed
        }
    }

    // Map the SHM BAR range into the guest process.
    let pfn = (nf.vndev.shm_phys + shm_offset) >> PAGE_SHIFT;
    remap_pfn_range(vma, vma.start, pfn, requested_len, vma.page_prot)?;

    Ok(())
}
```

This is why the gVisor KVM memory-type bug does not apply to us: gVisor
KVM sets all guest page table entries to write-back and relies on EPT to
fix caching. Here, the guest kernel module sets `pgprot` directly.

### 4.5 UVM Init Mapping

CUDA runtime initialization unconditionally maps ~2 MB of
`/dev/nvidia-uvm` at a fixed virtual address (e.g. `0x205000000`). In
gVisor KVM, this conflicts with the sentry's address space. In a real VM,
it does not conflict because:

- The guest has its own virtual address space and page tables.
- The guest kernel maps from the SHM BAR at whatever guest-physical
  address is convenient.
- The host-side mapping is at a VMM-chosen address, unrelated to the
  guest virtual address.

The guest driver handles this the same way as any other mmap, through
`nv_mmap()` above.

---

## 5. VMM Backend Design

### 5.1 State

```rust
use std::collections::HashMap;
use std::os::fd::RawFd;
use std::sync::Mutex;

struct NvBackend {
    /// Supported driver ABI for the detected host driver version.
    abi: DriverAbi,

    /// Maps guest handles to host file descriptors.
    handles: Mutex<HandleTable>,

    /// SHM region management.
    shm: ShmAllocator,
}

struct HandleTable {
    next_id: u32,
    map: HashMap<u32, HostHandle>,
}

struct HostHandle {
    fd: RawFd,
    dev_kind: DevKind,

    /// Mmap context, set by mapping ioctls like NV_ESC_RM_MAP_MEMORY.
    mmap_ctx: Option<MmapContext>,
}

struct MmapContext {
    shm_offset: u64,
    length: u64,
    mem_type: MemType,
}
```

### 5.2 Open / Close

```rust
impl NvBackend {
    fn handle_open(&self, req: &VirtioNvOpenReq) -> VirtioNvOpenResp {
        let path = match req.dev_kind {
            DevKind::Ctl => "/dev/nvidiactl".to_string(),
            DevKind::Gpu { index } => format!("/dev/nvidia{}", index),
            DevKind::Uvm => "/dev/nvidia-uvm".to_string(),
        };

        let fd = match unsafe {
            libc::open(path.as_ptr() as *const _, libc::O_RDWR)
        } {
            fd if fd >= 0 => fd,
            _ => return VirtioNvOpenResp { status: errno(), handle: 0 },
        };

        let mut handles = self.handles.lock().unwrap();
        let handle = handles.next_id;
        handles.next_id += 1;
        handles.map.insert(handle, HostHandle {
            fd,
            dev_kind: req.dev_kind,
            mmap_ctx: None,
        });

        VirtioNvOpenResp { status: 0, handle }
    }

    fn handle_close(&self, req: &VirtioNvCloseReq) -> VirtioNvCloseResp {
        let mut handles = self.handles.lock().unwrap();
        if let Some(h) = handles.map.remove(&req.handle) {
            // If there's an active mmap in the SHM region, unmap it.
            if let Some(ctx) = &h.mmap_ctx {
                self.shm.unmap(ctx.shm_offset, ctx.length);
            }
            unsafe { libc::close(h.fd); }
            VirtioNvCloseResp { status: 0 }
        } else {
            VirtioNvCloseResp { status: libc::EBADF }
        }
    }
}
```

### 5.3 Ioctl Dispatch

This is the core of the backend and where nvproxy's logic is ported. The
backend uses the `DriverAbi` to dispatch each ioctl number to the
appropriate handler:

```rust
struct DriverAbi {
    frontend_ioctl: HashMap<u32, FrontendIoctlHandler>,
    uvm_ioctl: HashMap<u32, UvmIoctlHandler>,
    control_cmd: HashMap<u32, ControlCmdHandler>,
    allocation_class: HashMap<u32, AllocationClassHandler>,
}

type FrontendIoctlHandler = fn(
    backend: &NvBackend,
    host_fd: RawFd,
    params: &mut [u8],
    handles: &mut HandleTable,
) -> IoctlResult;

struct IoctlResult {
    ret: i32,
    mmap_info: Option<MmapInfo>,
}

struct MmapInfo {
    shm_offset: u64,
    length: u64,
    mem_type: MemType,
}
```

Dispatch:

```rust
impl NvBackend {
    fn handle_ioctl(&self, req: &VirtioNvIoctlReq) -> VirtioNvIoctlResp {
        let mut handles = self.handles.lock().unwrap();
        let host_handle = match handles.map.get(&req.handle) {
            Some(h) => h,
            None => return VirtioNvIoctlResp::error(libc::EBADF),
        };

        let handler = match host_handle.dev_kind {
            DevKind::Ctl | DevKind::Gpu { .. } => {
                self.abi.frontend_ioctl.get(&req.nr)
            }
            DevKind::Uvm => {
                self.abi.uvm_ioctl.get(&req.nr)
            }
        };

        match handler {
            Some(func) => {
                let mut params = req.params.clone();
                let result = func(self, host_handle.fd, &mut params, &mut handles);
                VirtioNvIoctlResp {
                    ret: result.ret,
                    param_size: params.len() as u32,
                    params,
                    mmap_info: result.mmap_info,
                }
            }
            None => {
                log::warn!(
                    "virtio-gpu-nv: unhandled ioctl nr=0x{:x} for {:?}",
                    req.nr, host_handle.dev_kind
                );
                VirtioNvIoctlResp::error(libc::EINVAL)
            }
        }
    }
}
```

### 5.4 Ioctl Handler Categories

These categories mirror nvproxy exactly:

**Simple ioctls** — no embedded pointers or FDs:

```rust
fn ioctl_simple<T: NvIoctlParams>(
    _backend: &NvBackend,
    host_fd: RawFd,
    params: &mut [u8],
    _handles: &mut HandleTable,
) -> IoctlResult {
    // params is the raw ioctl struct, pass it through directly.
    let ret = unsafe {
        libc::ioctl(host_fd, T::CMD, params.as_mut_ptr())
    };
    IoctlResult {
        ret: if ret < 0 { -errno() } else { ret as i32 },
        mmap_info: None,
    }
}
```

**FD-carrying ioctls** — contain a guest FD that must be translated to a
host FD:

```rust
fn ioctl_has_fd<T: NvIoctlWithFd>(
    _backend: &NvBackend,
    host_fd: RawFd,
    params: &mut [u8],
    handles: &mut HandleTable,
) -> IoctlResult {
    let ioctl_params = T::from_bytes_mut(params);

    // Save the guest handle, replace with host FD.
    let guest_handle = ioctl_params.get_embedded_handle();
    let embedded_host = match handles.map.get(&guest_handle) {
        Some(h) => h.fd,
        None => return IoctlResult::error(libc::EBADF),
    };
    ioctl_params.set_embedded_fd(embedded_host);

    let ret = unsafe {
        libc::ioctl(host_fd, T::CMD, params.as_mut_ptr())
    };

    // Restore the guest handle before sending params back.
    ioctl_params.set_embedded_fd(guest_handle as i32);

    IoctlResult {
        ret: if ret < 0 { -errno() } else { ret as i32 },
        mmap_info: None,
    }
}
```

**Mapping ioctls** — set up an mmap context (e.g. `NV_ESC_RM_MAP_MEMORY`):

```rust
fn handle_rm_map_memory(
    backend: &NvBackend,
    host_fd: RawFd,
    params: &mut [u8],
    handles: &mut HandleTable,
) -> IoctlResult {
    let ioctl_params = IoctlNVOS33ParametersWithFD::from_bytes_mut(params);

    // Translate the embedded "map FD" handle.
    let map_guest_handle = ioctl_params.fd as u32;
    let map_host = match handles.map.get_mut(&map_guest_handle) {
        Some(h) => h,
        None => return IoctlResult::error(libc::EBADF),
    };
    let orig_fd = ioctl_params.fd;
    ioctl_params.fd = map_host.fd;

    // Forward to host driver.
    let ret = unsafe {
        libc::ioctl(host_fd, NV_ESC_RM_MAP_MEMORY_CMD, params.as_mut_ptr())
    };

    if ret >= 0 && ioctl_params.params.status == NV_OK {
        let length = ioctl_params.params.length;
        let caching_flags = ioctl_params.params.flags;
        let mem_type = get_memory_type(caching_flags);

        // Allocate a region in the SHM BAR.
        let shm_offset = backend.shm.allocate(length);

        // mmap the host device FD into the SHM region at that offset.
        // This is the same operation nvproxy does in MapInternal().
        let shm_host_addr = backend.shm.base_addr() + shm_offset as usize;
        unsafe {
            libc::mmap(
                shm_host_addr as *mut _,
                length as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_FIXED,
                map_host.fd,
                0,  // NVIDIA requires offset 0
            );
        }

        // Record the mapping context.
        map_host.mmap_ctx = Some(MmapContext {
            shm_offset,
            length,
            mem_type,
        });

        // Restore original FD before returning params to guest.
        ioctl_params.fd = orig_fd;

        IoctlResult {
            ret: 0,
            mmap_info: Some(MmapInfo { shm_offset, length, mem_type }),
        }
    } else {
        ioctl_params.fd = orig_fd;
        IoctlResult {
            ret: if ret < 0 { -errno() } else { 0 },
            mmap_info: None,
        }
    }
}
```

**Nested dispatch** — `NV_ESC_RM_CONTROL` and `NV_ESC_RM_ALLOC` perform
a second level of dispatch based on a command or class ID within the
ioctl parameters, exactly as nvproxy does:

```rust
fn handle_rm_control(
    backend: &NvBackend,
    host_fd: RawFd,
    params: &mut [u8],
    handles: &mut HandleTable,
) -> IoctlResult {
    let ioctl_params = NVOS54Parameters::from_bytes_mut(params);
    let cmd = ioctl_params.cmd;

    // Second-level dispatch by control command.
    match backend.abi.control_cmd.get(&cmd) {
        Some(handler) => handler(backend, host_fd, params, handles),
        None => {
            // If we don't recognize the command, try simple passthrough.
            // Many RM control commands have no embedded pointers.
            ioctl_simple_no_status(backend, host_fd, params, handles)
        }
    }
}
```

### 5.5 SHM Region Allocator

```rust
struct ShmAllocator {
    base: *mut u8,       // host virtual address of SHM region
    size: u64,
    // Simple allocator. Could be a more sophisticated one later.
    allocations: Mutex<Vec<ShmAllocation>>,
    next_offset: Mutex<u64>,
}

struct ShmAllocation {
    offset: u64,
    length: u64,
}

impl ShmAllocator {
    fn allocate(&self, length: u64) -> u64 {
        let aligned_length = page_align_up(length);
        let mut next = self.next_offset.lock().unwrap();
        let offset = *next;
        assert!(offset + aligned_length <= self.size,
                "SHM region exhausted");
        *next += aligned_length;
        self.allocations.lock().unwrap().push(ShmAllocation {
            offset,
            length: aligned_length,
        });
        offset
    }

    fn unmap(&self, offset: u64, length: u64) {
        let addr = unsafe { self.base.add(offset as usize) };
        unsafe {
            libc::munmap(addr as *mut _, page_align_up(length) as usize);
        }
        // Mark region as free (TODO: proper free-list for reuse).
    }

    fn base_addr(&self) -> usize {
        self.base as usize
    }
}
```

### 5.6 Memory Type Mapping

Ported from nvproxy's `getMemoryType`:

```rust
fn get_memory_type(caching_flags: u32) -> MemType {
    let caching_type = (caching_flags >> NVOS33_FLAGS_CACHING_TYPE_SHIFT)
        & NVOS33_FLAGS_CACHING_TYPE_MASK;

    match caching_type {
        NVOS33_FLAGS_CACHING_TYPE_CACHED
        | NVOS33_FLAGS_CACHING_TYPE_WRITEBACK => MemType::WriteBack,

        NVOS33_FLAGS_CACHING_TYPE_WRITECOMBINED
        | NVOS33_FLAGS_CACHING_TYPE_DEFAULT => MemType::WriteCombine,

        NVOS33_FLAGS_CACHING_TYPE_UNCACHED
        | NVOS33_FLAGS_CACHING_TYPE_UNCACHED_WEAK => MemType::Uncached,

        _ => {
            log::warn!("virtio-gpu-nv: unknown caching type {}", caching_type);
            MemType::Uncached
        }
    }
}
```

---

## 6. ABI Versioning

The NVIDIA kernel driver ABI is **not stable** between releases. Ioctl
struct layouts can change arbitrarily. nvproxy handles this with a
versioned ABI table; we do the same.

```rust
use std::collections::HashMap;

/// Each supported driver version has a corresponding ABI definition.
static SUPPORTED_ABIS: LazyLock<HashMap<DriverVersion, DriverAbi>> =
    LazyLock::new(|| {
        let mut m = HashMap::new();

        // Base version — defines all known ioctls for this release.
        m.insert(
            DriverVersion::new(535, 129, 3),
            build_abi_535_129_03(),
        );

        // Later versions override/extend specific handlers.
        m.insert(
            DriverVersion::new(550, 54, 15),
            build_abi_550_54_15(),
        );

        m
    });

/// On startup, probe the host driver version and select the matching ABI.
fn select_abi() -> Result<&'static DriverAbi> {
    let host_version = probe_host_driver_version()?;
    SUPPORTED_ABIS.get(&host_version).ok_or_else(|| {
        anyhow!("unsupported NVIDIA driver version: {}", host_version)
    })
}
```

When a new NVIDIA driver ships, adding support means:

1. Checking which ioctl structs changed (diffing against the open kernel
   modules repo).
2. Updating or adding struct definitions.
3. Adding a new entry in `SUPPORTED_ABIS`.

This is the same maintenance burden as nvproxy. Their changes can be
followed directly.

---

## 7. CUDA and NVENC Integration

### 7.1 Why CUDA Is Needed

The target use case includes a streaming pipeline:

```text
Guest compositor renders frame (Vulkan/GL)
  │
  │ zero-copy
  ▼
CUDA imports rendered image (cuGraphicsMapResources)
  │ GPU-side pointer, no CPU copy
  ▼
NVENC encodes directly from GPU memory
  │
  ▼
~100 KB compressed bitstream → cuMemcpyDtoH → send to client
```

Without CUDA, the compositor cannot bridge between the rendering output
and NVENC's input without a CPU-side copy of the full frame.

### 7.2 What CUDA Operations This Requires

All of these go through `/dev/nvidiactl` and `/dev/nvidia0`:

```text
cuInit()                            → RM ioctl
cuDeviceGet(), cuCtxCreate()        → RM alloc ioctls
cuMemAlloc() (device memory)        → RM alloc ioctl
cuGraphicsGLRegisterImage()         → RM control ioctl
cuGraphicsVulkanImportSemaphore()   → RM control ioctl
cuGraphicsMapResources()            → RM control ioctl
cuGraphicsResourceGetMappedPointer()→ returns CUdeviceptr (GPU VA)
NvEncOpenEncodeSession()            → RM ioctls
NvEncRegisterResource(CUdeviceptr)  → RM ioctls
NvEncEncodePicture()                → RM ioctls
cuMemcpyDtoH(bitstream)             → RM ioctl (small copy)
```

None of these use `cudaMallocManaged()` or UVM page-fault-driven
migration. The GPU virtual address (`CUdeviceptr`) is managed entirely
within the NVIDIA kernel driver; CPU addresses are not involved in the
rendering-to-encoding path.

### 7.3 The UVM Init Mapping

CUDA's `cuInit()` maps ~2 MB of `/dev/nvidia-uvm` at a fixed virtual
address. In gVisor KVM this can conflict with the sentry's address space.
In our VM it works because the guest has its own page tables and virtual
address space, independent of the VMM. See Section 4.5.

### 7.4 What Is Explicitly Out of Scope

- `cudaMallocManaged()`: requires UVM page fault handling across the VM
  boundary and precise virtual address matching. Known to be flaky even
  in gVisor KVM. Not needed for the NVENC pipeline.
- Pageable memory with automatic migration.
- Multi-GPU P2P transfers.

---

## 8. Graphics Integration

### 8.1 Vulkan

The guest runs `libvulkan_nvidia.so` unmodified. All Vulkan API calls
(draw, dispatch, pipeline creation, etc.) are handled entirely within the
guest by NVIDIA's user-mode driver. Only ioctls to `/dev/nvidiactl` and
`/dev/nvidia0` cross the VM boundary. This means:

- Command buffers are built locally in the guest (no serialization).
- Only queue submissions and resource management ioctls go through
  virtio.
- This is the same model as DRM native context on Intel/AMD.

### 8.2 OpenGL

The guest runs NVIDIA's `libGL_nvidia.so` / `libEGL_nvidia.so`. For
headless/offscreen rendering:

- Use `EGL_EXT_platform_device` (no display server needed).
- Or render to an X11 window via the guest's X11 proxy, using NVIDIA's
  GLX fallback path (X11 SHM for presentation when DRM is absent).

Because ioctl forwarding gives the guest a real driver stack, OpenGL
buffer tracking and implicit synchronization work correctly — unlike
Venus, where the host owns buffers and the guest cannot track them.

### 8.3 Compositor Buffer Sharing

The guest compositor is the Wayland (or X11) server. It receives buffers
from applications and composites them. Options for buffer sharing between
apps and compositor, without `/dev/nvidia-drm`:

- **EGLStreams**: NVIDIA's buffer-sharing mechanism. Uses `/dev/nvidiactl`
  only. Zero-copy between producer (app) and consumer (compositor).
- **Explicit CUDA import**: compositor imports app buffers via
  `cuGraphicsGLRegisterImage` or Vulkan external memory.
- **wl_shm**: CPU-shared buffers. Works universally but requires GPU
  upload in the compositor. Acceptable for non-GPU-accelerated apps.

### 8.4 Frame Presentation for Streaming

The compositor's output path:

```text
Compositor composites all windows into one image (GPU)
  │
  ├─ CUDA interop: get CUdeviceptr for composed image
  │
  ├─ NVENC: encode from CUdeviceptr (zero-copy on GPU)
  │
  ├─ cuMemcpyDtoH: read ~100KB encoded bitstream to CPU
  │
  └─ Send bitstream to client (vsock, network, SHM, etc.)
```

No DRM, no modeset, no physical display involvement. The entire render →
composite → encode pipeline runs on the GPU inside the guest.

---

## 9. Limitations

These are known, accepted limitations of the initial design:

- **NVIDIA-only**: this is not a vendor-agnostic solution. It proxies
  NVIDIA's specific kernel driver ABI.

- **Version-locked**: each supported NVIDIA driver version must be
  explicitly added with its ABI definitions.

- **Security surface**: forwarded ioctls go directly to the host NVIDIA
  kernel module. Bugs in those code paths are reachable from the guest.
  This is the same trade-off as nvproxy and VFIO passthrough.

- **No cudaMallocManaged**: full unified virtual memory is not supported.
  CUDA device memory allocations and graphics interop work.

- **No /dev/nvidia-drm or /dev/nvidia-modeset**: no physical display
  output from the guest. Design targets offscreen rendering + streaming.

- **Single GPU**: initial design assumes one NVIDIA GPU on the host. No
  MIG or SR-IOV support.

- **SHM region sizing**: the SHM BAR must be sized at VM creation time.
  If GPU workloads need more mmap space than was provisioned, allocations
  will fail.

---

## 10. Relationship to Prior Work

### gVisor nvproxy

virtio-gpu-nv is directly inspired by nvproxy. The ABI definitions,
ioctl handler categories, and overall ioctl-forwarding strategy are
designed to be ported from nvproxy's Go implementation to Rust. The key
structural difference is the transport: nvproxy uses in-process syscall
interception; we use virtio over KVM.

### WSL2 /dev/dxg

Microsoft and NVIDIA built a similar system for WSL2 using Hyper-V's
VMBus and a custom `dxgkrnl` guest kernel module (~30K lines of C). This
proves the concept works for a production system. Our approach is
smaller in scope (Linux-to-Linux, open ecosystem, no DirectX) but
architecturally similar.

### DRM native context (Intel/AMD)

DRM native context achieves the same goal (guest runs real driver, builds
command buffers locally, only submits cross the VM boundary) for
Intel/AMD GPUs via virtio-gpu. virtio-gpu-nv aims to provide equivalent
capabilities for NVIDIA.
