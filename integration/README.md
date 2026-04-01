# libkrun integration

## Setup

1. Fork libkrun
2. Add dependency in `libkrun/Cargo.toml` or the devices crate:
   ```toml
   [dependencies]
   virtio-gpu-nv-device = { path = "../virtio-gpu-nv/crates/device" }
