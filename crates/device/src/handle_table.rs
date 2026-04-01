// crates/device/src/handle_table.rs
//
// Maps opaque 64-bit guest handles to open host file descriptors.
//
// The backend generates a new handle each time the guest issues an OPEN
// request.  Handles are never reused (monotonically increasing counter)
// so a stale handle from a crashed guest can't accidentally alias a new one.

use std::collections::HashMap;
use std::os::fd::{OwnedFd, RawFd, AsRawFd};

use crate::error::{DeviceError, Result};

pub struct HandleTable {
    /// Next handle value to issue.
    next: u64,
    /// Maps guest handle → owned host fd.
    table: HashMap<u64, OwnedFd>,
}

impl HandleTable {
    pub fn new() -> Self {
        Self {
            next: 1,   // 0 is reserved as the null/invalid handle
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

    /// Remove and close the fd associated with `handle`.
    pub fn remove(&mut self, handle: u64) -> Result<()> {
        self.table
            .remove(&handle)
            .map(|_| ())
            .ok_or(DeviceError::BadHandle(handle))
    }

    /// Number of open handles (for tests / diagnostics).
    pub fn len(&self) -> usize {
        self.table.len()
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
        // Open /dev/null as a harmless real fd for testing.
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
}
