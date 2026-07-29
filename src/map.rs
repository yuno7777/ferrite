//! Read-only memory mapping, hand-rolled.
//!
//! No `memmap2`, no `libc` — just the two syscalls, declared inline. A 4 GB
//! model must not become a 4 GB `Vec`, so this is not optional even in a
//! dependency-free crate.

use std::fs::File;
use std::io::{Error, ErrorKind, Result};
use std::path::Path;

#[cfg(not(any(windows, unix)))]
compile_error!("ferrite needs mmap: supported on windows and unix only");

pub struct Mmap {
    ptr: *const u8,
    len: usize,
    #[cfg(windows)]
    mapping: *mut core::ffi::c_void,
}

// The mapping is read-only and never reallocated, so sharing it across threads
// is sound. Threaded decode depends on this.
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        if len == 0 {
            return Err(Error::new(ErrorKind::InvalidData, "file is empty"));
        }
        let len = usize::try_from(len)
            .map_err(|_| Error::new(ErrorKind::InvalidData, "file exceeds address space"))?;
        map(&file, len)
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // Valid for `len` bytes for as long as `self` lives; the file mapping
        // keeps the pages alive even after `File` is dropped.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(windows)]
mod sys {
    use core::ffi::c_void;

    pub type Handle = *mut c_void;
    pub const PAGE_READONLY: u32 = 0x02;
    pub const FILE_MAP_READ: u32 = 0x04;

    unsafe extern "system" {
        pub fn CreateFileMappingW(
            file: Handle,
            attrs: *mut c_void,
            protect: u32,
            max_size_high: u32,
            max_size_low: u32,
            name: *const u16,
        ) -> Handle;
        pub fn MapViewOfFile(
            mapping: Handle,
            access: u32,
            offset_high: u32,
            offset_low: u32,
            bytes: usize,
        ) -> *mut c_void;
        pub fn UnmapViewOfFile(base: *const c_void) -> i32;
        pub fn CloseHandle(handle: Handle) -> i32;
    }
}

#[cfg(windows)]
fn map(file: &File, len: usize) -> Result<Mmap> {
    use std::os::windows::io::AsRawHandle;

    let handle = file.as_raw_handle() as sys::Handle;
    // Size 0/0 means "whole file", so the length never has to be split into
    // high/low words here.
    let mapping = unsafe {
        sys::CreateFileMappingW(
            handle,
            std::ptr::null_mut(),
            sys::PAGE_READONLY,
            0,
            0,
            std::ptr::null(),
        )
    };
    if mapping.is_null() {
        return Err(Error::last_os_error());
    }
    let base = unsafe { sys::MapViewOfFile(mapping, sys::FILE_MAP_READ, 0, 0, 0) };
    if base.is_null() {
        let err = Error::last_os_error();
        unsafe { sys::CloseHandle(mapping) };
        return Err(err);
    }
    Ok(Mmap {
        ptr: base as *const u8,
        len,
        mapping,
    })
}

#[cfg(unix)]
mod sys {
    use core::ffi::c_void;

    pub const PROT_READ: i32 = 1;
    pub const MAP_PRIVATE: i32 = 2;

    unsafe extern "C" {
        // off_t is i64 on every 64-bit target we support.
        pub fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> i32;
    }
}

#[cfg(unix)]
fn map(file: &File, len: usize) -> Result<Mmap> {
    use std::os::unix::io::AsRawFd;

    let ptr = unsafe {
        sys::mmap(
            std::ptr::null_mut(),
            len,
            sys::PROT_READ,
            sys::MAP_PRIVATE,
            file.as_raw_fd(),
            0,
        )
    };
    if ptr == usize::MAX as *mut core::ffi::c_void {
        return Err(Error::last_os_error());
    }
    Ok(Mmap {
        ptr: ptr as *const u8,
        len,
    })
}

impl Drop for Mmap {
    fn drop(&mut self) {
        #[cfg(windows)]
        unsafe {
            sys::UnmapViewOfFile(self.ptr as *const core::ffi::c_void);
            sys::CloseHandle(self.mapping);
        }
        #[cfg(unix)]
        unsafe {
            sys::munmap(self.ptr as *mut core::ffi::c_void, self.len);
        }
    }
}
