// crates/device/src/handle_table.rs
//
// Maps opaque 64-bit guest handles to open host file descriptors.
//
// --- Handle lifecycle and ungraceful teardown ---
//
// Normal path: guest calls close() → nv_release() → NV_MSG_CLOSE →
//   HandleTable::remove() drops the OwnedFd, which closes the host fd.
//   The host NVIDIA driver sees the close() and frees its internal state.
//
// Ungraceful path (VM crash / process kill / SIGKILL):
//   The virtio device is reset (virtio_reset_device() in nv_remove()).
//   nv_remove() calls NvidiaBackend::teardown(), which calls
//   HandleTable::drain_all().  drain_all() drops every OwnedFd, which
//   closes every host fd in one sweep.  The host NVIDIA driver's fd-release
//   path runs for each one, freeing all RM objects associated with that fd.
//
//   This is analogous to nvproxy's Release() in
//   pkg/sentry/devices/nvproxy/nvproxy.go, which iterates all files and
//   calls their Close().
//
// NOTE: RM objects allocated via NV_ESC_RM_ALLOC are associated with the
// nvidiactl fd, not with individual /dev/nvidia# fds.  Closing nvidiactl
// is sufficient to free all RM objects for that client.  We close ALL fds
// in drain_all() anyway to be safe.

use std::collections::HashMap;
use std::os::fd::{AsRawFd, OwnedFd, RawFd};

use crate::error::{DeviceError, Result};

pub struct HandleTable {
    /// Next handle value to issue.  Monotonically increasing so a stale
    /// handle from a crashed guest cannot alias a new handle.
    next: u64,
    /// Maps guest handle → owned host fd.
    /// OwnedFd's Drop impl closes the fd when the entry is removed.
    table: HashMap<u64, OwnedFd>,
}

impl HandleTable {
    pub fn new() -> Self {
        Self {
            next: 1, // 0 is reserved as null/invalid
            table: HashMap::new(),
        }
    }

    /// Insert an owned fd and return the new guest handle.
    pub fn insert(&mut self, fd: OwnedFd) -> u64 {
        let handle = self.next;
        self.next += 1;
        self.table.insert(handle, fd);
        handle
    }

    /// Borrow the raw fd associated with `handle`.
    pub fn get_raw(&self, handle: u64) -> Result<RawFd> {
        self.table
            .get(&handle)
            .map(|fd| fd.as_raw_fd())
            .ok_or(DeviceError::BadHandle(handle))
    }

    /// Remove and close the fd associated with `handle` (normal close path).
    pub fn remove(&mut self, handle: u64) -> Result<()> {
        self.table
            .remove(&handle)
            .map(|_| ())
            .ok_or(DeviceError::BadHandle(handle))
    }

    /// Close every open fd and empty the table (teardown / crash recovery).
    ///
    /// Dropping each OwnedFd calls close(2) on the host fd.  The host
    /// NVIDIA driver's release() path then frees all RM objects for that fd.
    ///
    /// After this call, len() == 0.  The table can be reused (e.g. after a
    /// VM reset) without reallocating.
    pub fn drain_all(&mut self) {
        let count = self.table.len();
        if count > 0 {
            tracing::info!("HandleTable::drain_all: closing {} host fds", count);
        }
        // HashMap::drain() drops each OwnedFd as it removes it.
        for (handle, fd) in self.table.drain() {
            tracing::debug!(
                "  closing guest_handle={} host_fd={}",
                handle,
                fd.as_raw_fd()
            );
            drop(fd); // explicit for clarity; drop(handle) is a no-op
        }
        // Reset the counter so post-teardown handles don't collide if the
        // backend is reused for a new VM (e.g. libkrun VM restart).
        self.next = 1;
    }

    /// Number of open handles.
    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }
}

impl Default for HandleTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::FromRawFd;

    fn make_fd() -> OwnedFd {
        let raw = unsafe { libc::open(c"/dev/null".as_ptr(), libc::O_RDONLY) };
        assert!(raw >= 0);
        unsafe { OwnedFd::from_raw_fd(raw) }
    }

    #[test]
    fn insert_get_remove() {
        let mut t = HandleTable::new();
        let h = t.insert(make_fd());
        assert!(h > 0);
        assert!(t.get_raw(h).is_ok());
        assert!(t.remove(h).is_ok());
        assert!(matches!(t.get_raw(h), Err(DeviceError::BadHandle(_))));
    }

    #[test]
    fn handles_are_unique() {
        let mut t = HandleTable::new();
        let h1 = t.insert(make_fd());
        let h2 = t.insert(make_fd());
        assert_ne!(h1, h2);
    }

    #[test]
    fn drain_all_empties_table() {
        let mut t = HandleTable::new();
        t.insert(make_fd());
        t.insert(make_fd());
        t.insert(make_fd());
        assert_eq!(t.len(), 3);
        t.drain_all();
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn drain_all_resets_counter() {
        let mut t = HandleTable::new();
        t.insert(make_fd()); // handle = 1
        t.insert(make_fd()); // handle = 2
        t.drain_all();
        // After drain, next handle should restart from 1
        let h = t.insert(make_fd());
        assert_eq!(h, 1, "counter should reset after drain_all");
    }

    #[test]
    fn stale_handle_after_drain_is_rejected() {
        let mut t = HandleTable::new();
        let h = t.insert(make_fd());
        t.drain_all();
        // The old handle is gone; get_raw must fail
        assert!(matches!(t.get_raw(h), Err(DeviceError::BadHandle(_))));
    }
}
