// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io;
#[cfg(all(not(unix), not(windows)))]
use std::io::Read;
#[cfg(all(not(unix), not(windows)))]
use std::io::Seek;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
#[cfg(target_os = "linux")]
use rustix::fs::AtFlags;
#[cfg(target_os = "linux")]
use rustix::fs::Mode;
#[cfg(target_os = "linux")]
use rustix::fs::OFlags;
#[cfg(target_os = "linux")]
use rustix::fs::StatxFlags;
use vortex_array::buffer::BufferHandle;
use vortex_array::memory::DefaultHostAllocator;
use vortex_array::memory::HostAllocatorRef;
use vortex_buffer::Alignment;
use vortex_error::VortexResult;
#[cfg(not(target_os = "linux"))]
use vortex_error::vortex_bail;
#[cfg(target_os = "linux")]
use vortex_error::vortex_ensure;
#[cfg(target_os = "linux")]
use vortex_error::vortex_err;

use crate::CoalesceConfig;
use crate::VortexReadAt;
use crate::runtime::Handle;

/// Read exactly `buffer.len()` bytes from `file` starting at `offset`.
/// This is a platform-specific helper that uses the most efficient method available.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        file.read_exact_at(buffer, offset)
    }
    #[cfg(windows)]
    {
        let mut bytes_read = 0;
        while bytes_read < buffer.len() {
            let read = file.seek_read(&mut buffer[bytes_read..], offset + bytes_read as u64)?;
            if read == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                ));
            }
            bytes_read += read;
        }
        Ok(())
    }
    #[cfg(all(not(unix), not(windows)))]
    {
        use std::io::SeekFrom;
        let mut file_ref = file;
        file_ref.seek(SeekFrom::Start(offset))?;
        file_ref.read_exact(buffer)
    }
}

/// Default number of concurrent requests to allow for local file I/O.
pub const DEFAULT_CONCURRENCY: usize = 32;

#[cfg(target_os = "linux")]
const FALLBACK_DIRECT_IO_ALIGNMENT: usize = 4096;

#[cfg(target_os = "linux")]
#[derive(Clone, Copy)]
struct DirectIoAlignment {
    memory: usize,
    offset: usize,
}

#[cfg(target_os = "linux")]
impl Default for DirectIoAlignment {
    fn default() -> Self {
        Self {
            memory: FALLBACK_DIRECT_IO_ALIGNMENT,
            offset: FALLBACK_DIRECT_IO_ALIGNMENT,
        }
    }
}

/// An adapter type wrapping a [`File`] to implement [`VortexReadAt`].
pub struct FileReadAt {
    uri: Arc<str>,
    file: Arc<File>,
    handle: Handle,
    allocator: HostAllocatorRef,
    #[cfg(target_os = "linux")]
    direct_io: bool,
    #[cfg(target_os = "linux")]
    direct_io_alignment: DirectIoAlignment,
}

impl FileReadAt {
    /// Open a file for reading.
    pub fn open(path: impl AsRef<Path>, handle: Handle) -> VortexResult<Self> {
        Self::open_with_allocator(path, handle, Arc::new(DefaultHostAllocator))
    }

    /// Open a file for reading using a custom writable buffer allocator.
    pub fn open_with_allocator(
        path: impl AsRef<Path>,
        handle: Handle,
        allocator: HostAllocatorRef,
    ) -> VortexResult<Self> {
        let path = path.as_ref();
        let uri = path.to_string_lossy().to_string().into();
        let file = Arc::new(File::open(path)?);
        Ok(Self {
            uri,
            file,
            handle,
            allocator,
            #[cfg(target_os = "linux")]
            direct_io: false,
            #[cfg(target_os = "linux")]
            direct_io_alignment: DirectIoAlignment::default(),
        })
    }

    /// Open a file for direct I/O, bypassing the operating system page cache.
    ///
    /// This option is supported only on Linux. Unaligned logical reads are widened to satisfy the
    /// filesystem's direct-I/O requirements and sliced back to the requested range.
    pub fn open_direct(path: impl AsRef<Path>, handle: Handle) -> VortexResult<Self> {
        Self::open_direct_with_allocator(path, handle, Arc::new(DefaultHostAllocator))
    }

    /// Open a file for direct I/O using a custom writable buffer allocator.
    ///
    /// The allocator must honor the requested alignment. This option is supported only on Linux.
    #[cfg(target_os = "linux")]
    pub fn open_direct_with_allocator(
        path: impl AsRef<Path>,
        handle: Handle,
        allocator: HostAllocatorRef,
    ) -> VortexResult<Self> {
        let path = path.as_ref();
        let uri = path.to_string_lossy().to_string().into();
        let file = File::from(
            rustix::fs::open(
                path,
                OFlags::RDONLY | OFlags::CLOEXEC | OFlags::DIRECT,
                Mode::empty(),
            )
            .map_err(io::Error::from)?,
        );
        let direct_io_alignment = direct_io_alignment(&file);
        vortex_ensure!(
            direct_io_alignment.memory.is_power_of_two(),
            "direct I/O memory alignment must be a power of two, got {}",
            direct_io_alignment.memory
        );
        vortex_ensure!(
            direct_io_alignment.offset.is_power_of_two(),
            "direct I/O offset alignment must be a power of two, got {}",
            direct_io_alignment.offset
        );
        Ok(Self {
            uri,
            file: Arc::new(file),
            handle,
            allocator,
            direct_io: true,
            direct_io_alignment,
        })
    }

    /// Return an error when direct I/O is requested on an unsupported platform.
    #[cfg(not(target_os = "linux"))]
    pub fn open_direct_with_allocator(
        _path: impl AsRef<Path>,
        _handle: Handle,
        _allocator: HostAllocatorRef,
    ) -> VortexResult<Self> {
        vortex_bail!("direct file I/O is only supported on Linux")
    }
}

impl VortexReadAt for FileReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(CoalesceConfig::file())
    }

    fn concurrency(&self) -> usize {
        DEFAULT_CONCURRENCY
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let file = Arc::clone(&self.file);
        async move {
            let metadata = file.metadata()?;
            Ok(metadata.len())
        }
        .boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let file = Arc::clone(&self.file);
        let handle = self.handle.clone();
        let allocator = Arc::clone(&self.allocator);
        #[cfg(target_os = "linux")]
        let direct_io = self.direct_io;
        #[cfg(target_os = "linux")]
        let direct_io_alignment = self.direct_io_alignment;
        async move {
            handle
                .spawn_blocking(move || {
                    #[cfg(target_os = "linux")]
                    if direct_io {
                        let (read_offset, read_length, requested_range) =
                            direct_io_range(offset, length, direct_io_alignment.offset)?;
                        let allocation_alignment =
                            Alignment::new(direct_io_alignment.memory.max(*alignment));
                        let mut buffer = allocator.allocate(read_length, allocation_alignment)?;
                        let address = buffer.as_mut_slice().as_ptr() as usize;
                        vortex_ensure!(
                            address.is_multiple_of(direct_io_alignment.memory),
                            "host buffer address {address:#x} is not aligned to {} bytes",
                            direct_io_alignment.memory
                        );
                        read_direct_at(
                            &file,
                            buffer.as_mut_slice(),
                            read_offset,
                            requested_range.end,
                            direct_io_alignment,
                        )?;
                        return Ok(BufferHandle::new_host(
                            buffer.freeze().slice(requested_range),
                        ));
                    }

                    let mut buffer = allocator.allocate(length, alignment)?;
                    read_exact_at(&file, buffer.as_mut_slice(), offset)?;
                    Ok(BufferHandle::new_host(buffer.freeze()))
                })
                .await
        }
        .boxed()
    }
}

#[cfg(target_os = "linux")]
fn direct_io_range(
    offset: u64,
    length: usize,
    alignment: usize,
) -> VortexResult<(u64, usize, std::ops::Range<usize>)> {
    vortex_ensure!(alignment > 0, "direct I/O alignment must be non-zero");
    let alignment_u64 = u64::try_from(alignment)?;
    let length_u64 = u64::try_from(length)?;
    offset.checked_add(length_u64).ok_or_else(|| {
        vortex_err!("direct I/O range overflow: offset={offset}, length={length}")
    })?;
    let read_offset = offset / alignment_u64 * alignment_u64;
    let prefix = usize::try_from(offset - read_offset)?;
    let requested_end = prefix.checked_add(length).ok_or_else(|| {
        vortex_err!("direct I/O range overflow: offset={offset}, length={length}")
    })?;
    let read_length = requested_end
        .checked_add(alignment - 1)
        .ok_or_else(|| vortex_err!("direct I/O aligned length overflow"))?
        / alignment
        * alignment;

    Ok((read_offset, read_length, prefix..requested_end))
}

#[cfg(target_os = "linux")]
fn read_direct_at(
    file: &File,
    buffer: &mut [u8],
    offset: u64,
    required_bytes: usize,
    alignment: DirectIoAlignment,
) -> io::Result<usize> {
    let mut filled = 0;
    while filled < required_bytes {
        let filled_u64 = u64::try_from(filled)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "read offset overflow"))?;
        let read_offset = offset
            .checked_add(filled_u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "read offset overflow"))?;
        let bytes_read = match file.read_at(&mut buffer[filled..], read_offset) {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if bytes_read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "direct read returned {filled} bytes, but {required_bytes} bytes were required"
                ),
            ));
        }
        filled += bytes_read;
        if filled < required_bytes
            && (!filled.is_multiple_of(alignment.offset)
                || !filled.is_multiple_of(alignment.memory))
        {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "direct read returned an unaligned short read of {filled} bytes before the required {required_bytes} bytes"
                ),
            ));
        }
    }

    Ok(filled)
}

#[cfg(target_os = "linux")]
fn direct_io_alignment(file: &File) -> DirectIoAlignment {
    let Ok(stat) = rustix::fs::statx(
        file,
        c"",
        AtFlags::EMPTY_PATH | AtFlags::STATX_DONT_SYNC,
        StatxFlags::DIOALIGN,
    ) else {
        return DirectIoAlignment::default();
    };
    if stat.stx_mask & StatxFlags::DIOALIGN.bits() == 0 {
        return DirectIoAlignment::default();
    }

    let Ok(memory) = usize::try_from(stat.stx_dio_mem_align) else {
        return DirectIoAlignment::default();
    };
    let Ok(offset) = usize::try_from(stat.stx_dio_offset_align) else {
        return DirectIoAlignment::default();
    };
    if memory == 0 || offset == 0 {
        return DirectIoAlignment::default();
    }

    DirectIoAlignment { memory, offset }
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "linux")]
    use super::*;

    #[cfg(target_os = "linux")]
    #[test]
    fn widens_unaligned_direct_read_to_block_boundaries() -> VortexResult<()> {
        assert_eq!(direct_io_range(5, 10, 4096)?, (0, 4096, 5..15));
        assert_eq!(direct_io_range(4090, 20, 4096)?, (0, 8192, 4090..4110));
        assert_eq!(direct_io_range(4096, 4096, 4096)?, (4096, 4096, 0..4096));
        assert_eq!(direct_io_range(513, 1, 512)?, (512, 512, 1..2));
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn rejects_overflowing_direct_read_range() {
        assert!(direct_io_range(u64::MAX, 2, 4096).is_err());
    }
}
