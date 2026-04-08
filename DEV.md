# DEV.md — Local Development Guide

This document explains how to build every component and run Phase 1 & 2
tests on a Linux host with an NVIDIA GPU.

---

## 0. Prerequisites

```
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable

# Kernel headers for the guest driver (must match the kernel you'll boot)
sudo apt install linux-headers-$(uname -r) build-essential

# QEMU + KVM
sudo apt install qemu-system-x86 ovmf

# libkrun build dependencies (for Phase 3+)
sudo apt install libvirglrenderer-dev libepoxy-dev libgbm-dev \
                 libdrm-dev meson ninja-build

# Verify NVIDIA host driver is loaded
ls /dev/nvidia* && nvidia-smi
```

---

## 1. Building the Rust workspace (backend crates)

```bash
cd /path/to/virtio-gpu-nv

# Build everything
cargo build

# Run all unit tests (GPU-present tests auto-skip if /dev/nvidiactl absent)
cargo test

# Run only the backend tests with log output
RUST_LOG=debug cargo test -p device -- --nocapture
```

### What the tests cover (Phase 1 & 2)

| Test                              | GPU needed? | What it checks             |
| --------------------------------- | ----------- | -------------------------- |
| `open_invalid_gpu_index`          | No          | error path, no panic       |
| `close_unknown_handle`            | No          | BadHandle status           |
| `short_request_rejected`          | No          | bounds check               |
| `open_close_nvidiactl`            | Yes         | real open/close round-trip |
| `check_version_str`               | Yes         | full simple-ioctl path     |
| `handle_table::insert_get_remove` | No          | handle lifecycle           |
| `shm::basic_alloc`                | No          | bump allocator             |
| `version::parse_ok`               | No          | version string parsing     |

---

## 2. Building the guest kernel module

The module is compiled **inside the guest VM** against the guest kernel's
build tree. For local testing you can build and `insmod` it on the host
if the host kernel matches — but the char devices will conflict with the
real NVIDIA driver, so use a dedicated test VM (see §4).

```bash
cd guest-driver

# Against the currently-running kernel (host or guest)
make KDIR=/lib/modules/$(uname -r)/build

# Output
ls virtio_gpu_nv.ko
```

---

## 3. libkrun integration (Phase 3+ — read-only for Phase 2)

`virtio-gpu-nv` is a **library** that libkrun depends on. We do not fork
or embed libkrun. The integration point is:

```toml
# In libkrun's Cargo.toml:
[dependencies]
virtio-gpu-nv-device = { path = "../virtio-gpu-nv/crates/device" }
```

For Phase 2 testing we skip libkrun entirely and use a minimal QEMU-based
test harness (§4) that implements the virtio transport directly.

---

## 4. Running an end-to-end Phase 2 test

### 4a. Architecture

```
┌─── Guest VM (QEMU) ─────────────────────────────────────────┐
│  /dev/nvidiactl  ←── virtio_gpu_nv.ko ──► virtqueue          │
└──────────────────────────────────────────────────────────────┘
         ▲ virtio transport (vhost-user socket)
┌─── Host process ────────────────────────────────────────────┐
│  test-harness   →  NvidiaBackend::dispatch()                 │
│                 →  open("/dev/nvidiactl") on host            │
│                 →  ioctl(host_fd, ...)                       │
└──────────────────────────────────────────────────────────────┘
```

For Phase 2, we use **vhost-user** to connect QEMU's virtio device
frontend to our backend process over a Unix socket. This avoids
modifying libkrun for the test loop.

### 4b. Step-by-step

**Step 1 — Build the test harness binary**

```bash
# The test harness lives in crates/device/src/bin/test_harness.rs
# (created below).  It implements a minimal vhost-user backend.

cargo build --bin test-harness
```

**Step 2 — Prepare a guest disk image**

```bash
# Download a minimal cloud image (Ubuntu 22.04 or similar)
wget https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img
qemu-img convert -O raw jammy-server-cloudimg-amd64.img guest.raw
qemu-img resize guest.raw +10G

# Build the guest kernel module inside the image:
# Boot once with the standard virtio-net, copy the module, insmod it.
# (See Step 5 for a faster approach using 9p virtio-fs.)
```

**Step 3 — Start the test harness**

```bash
RUST_LOG=debug ./target/debug/test-harness \
    --socket /tmp/nv-vhost.sock \
    --shm-size $((256 * 1024 * 1024))
```

The harness listens on the socket, opens real `/dev/nvidia*` handles on
the host, and forwards ioctls using `NvidiaBackend::dispatch()`.

**Step 4 — Boot the guest VM**

```bash
qemu-system-x86_64 \
    -enable-kvm \
    -m 4G \
    -smp 4 \
    -cpu host \
    -drive file=guest.raw,format=raw,if=virtio \
    -chardev socket,id=nv,path=/tmp/nv-vhost.sock \
    -device vhost-user-blk-pci,chardev=nv,id=nv0 \
    -object memory-backend-memfd,id=mem,size=4G,share=on \
    -numa node,memdev=mem \
    -nographic
```

> **Note:** The vhost-user device type will change to a custom virtio
> device once libkrun integration lands. For Phase 2 testing, any
> vhost-user transport that gives us a virtqueue pair is sufficient.

**Step 5 — Inside the guest: load the module**

```bash
# Copy the .ko into the guest (via virtio-9p or scp)
sudo insmod virtio_gpu_nv.ko

# Verify the char devices appeared
ls -la /dev/nvidia*
# Expected:
#   /dev/nvidiactl
#   /dev/nvidia0
#   /dev/nvidia-uvm

# Check dmesg
dmesg | grep virtio-gpu-nv
```

**Step 6 — Run the Phase 2 smoke test**

```c
// test_phase2.c — compile inside the guest
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

// NV_ESC_CHECK_VERSION_STR parameter struct
struct nv_check_version {
    char   version[64];
    unsigned int reply;
};

// _IOWR('F', 0x25, struct nv_check_version)
#define NV_IOCTL_MAGIC 'F'
#define NV_ESC_CHECK_VERSION_STR 0x25
#define NV_CHECK_VERSION_STR \
    _IOWR(NV_IOCTL_MAGIC, NV_ESC_CHECK_VERSION_STR, struct nv_check_version)

int main(void) {
    int fd = open("/dev/nvidiactl", O_RDWR);
    if (fd < 0) { perror("open"); return 1; }
    printf("open() OK, fd=%d\n", fd);

    struct nv_check_version cv;
    memset(&cv, 0, sizeof(cv));
    // Leave version[0] = '\0': driver will fill it in and set reply=1 on match.

    int rc = ioctl(fd, NV_CHECK_VERSION_STR, &cv);
    printf("ioctl rc=%d  version=\"%s\"  reply=%u\n",
           rc, cv.version, cv.reply);

    close(fd);
    printf("close() OK\n");
    return (rc == 0) ? 0 : 1;
}
```

```bash
# Inside the guest:
gcc -o test_phase2 test_phase2.c
./test_phase2
# Expected output:
#   open() OK, fd=3
#   ioctl rc=0  version="535.129.03"  reply=1
#   close() OK
```

---

## 5. Iterating quickly without a full VM

For rapid backend iteration (no kernel module needed):

```bash
# Run the backend unit tests directly on the host.
# Tests that touch /dev/nvidiactl run only if the GPU is present.
cargo test -p device -- --nocapture 2>&1 | grep -E '(PASS|FAIL|test |ok|SKIP)'
```

For testing the guest driver in isolation (no NVIDIA GPU needed):

```bash
# Use a loopback virtio-test harness: the backend replies to every
# ioctl with Status::Ok and a zeroed param buffer.
# This lets you insmod the .ko and run open()/ioctl()/close() without
# real GPU hardware.
cargo run --bin test-harness -- --mock
```

---

## 6. Phase roadmap recap

| Phase | What gets tested                                 | GPU required  |
| ----- | ------------------------------------------------ | ------------- |
| 1     | `open()` + `close()` of `/dev/nvidiactl`         | Yes (or mock) |
| 2     | Simple ioctls (`NV_ESC_CHECK_VERSION_STR`, etc.) | Yes           |
| 3     | `NV_ESC_RM_ALLOC`, `NV_ESC_RM_CONTROL`, SHM mmap | Yes           |
| 4     | CUDA init, `/dev/nvidia-uvm`                     | Yes           |
| 5     | Full Vulkan (`vkcube`)                           | Yes           |

---

## 7. Troubleshooting

**`insmod` fails: "unknown symbol"**
Kernel version mismatch between where the module was built and where
it's being loaded. Rebuild with `KDIR` pointing at the running guest
kernel's build directory.

**`open("/dev/nvidiactl")` returns `ENODEV` in guest**
The virtio device wasn't found. Check that:

- The test harness is running and listening on the socket before QEMU starts.
- QEMU's `-device` line matches the socket path.
- `dmesg | grep virtio` shows the device being probed.

**ioctl returns `ENOTTY`**
The escape number is not in the dispatch table yet (Phase 3 ioctls).
This is expected for `NV_ESC_RM_ALLOC` etc. in Phase 2.

**ioctl returns `ENOSYS`**
The ioctl hit a Phase 3/4 stub in the backend. Same as above.

**Version string is empty / `reply=0`**
The host NVIDIA driver version does not match the ABI table in
`crates/abi/src/versions/`. Add support for the installed version
by following the pattern in `v535_129_03.rs`.
