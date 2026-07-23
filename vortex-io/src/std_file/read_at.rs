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
#[cfg(any(target_os = "linux", test))]
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
#[cfg(any(target_os = "linux", test))]
use vortex_error::vortex_ensure;
#[cfg(any(target_os = "linux", test))]
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
/// Conservative direct-I/O alignment used when Linux cannot report the filesystem constraints.
///
/// A page-sized fallback is accepted by common block devices and filesystems. If the actual
/// requirement is stricter, the read fails with the underlying `EINVAL`.
const FALLBACK_DIRECT_IO_ALIGNMENT: usize = 4096;

#[cfg(target_os = "linux")]
#[derive(Clone, Copy)]
struct DirectIoConstraints {
    /// Required alignment of the address of the userspace I/O buffer.
    memory_alignment: usize,
    /// Required alignment of both the file offset and the I/O length.
    offset_alignment: usize,
}

#[cfg(target_os = "linux")]
impl Default for DirectIoConstraints {
    fn default() -> Self {
        Self {
            memory_alignment: FALLBACK_DIRECT_IO_ALIGNMENT,
            offset_alignment: FALLBACK_DIRECT_IO_ALIGNMENT,
        }
    }
}

#[cfg(any(target_os = "linux", test))]
#[derive(Debug, PartialEq, Eq)]
struct DirectIoRange {
    read_offset: u64,
    read_length: usize,
    requested_range: std::ops::Range<usize>,
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
    direct_io_constraints: DirectIoConstraints,
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
            direct_io_constraints: DirectIoConstraints::default(),
        })
    }

    /// Open a file for direct I/O, bypassing the operating system page cache.
    ///
    /// This option is supported only on Linux. Unaligned logical reads are widened to satisfy the
    /// filesystem's direct-I/O requirements and sliced back to the requested range.
    #[cfg(target_os = "linux")]
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
        let direct_io_constraints = direct_io_constraints(&file)?;
        Ok(Self {
            uri,
            file: Arc::new(file),
            handle,
            allocator,
            direct_io: true,
            direct_io_constraints,
        })
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
        let direct_io_constraints = self.direct_io_constraints;
        async move {
            handle
                .spawn_blocking(move || {
                    #[cfg(target_os = "linux")]
                    if direct_io {
                        let direct_range = direct_io_range(
                            offset,
                            length,
                            direct_io_constraints.offset_alignment,
                        )?;
                        let allocation_alignment =
                            Alignment::new(direct_io_constraints.memory_alignment.max(*alignment));
                        let mut buffer =
                            allocator.allocate(direct_range.read_length, allocation_alignment)?;
                        let address = buffer.as_mut_slice().as_ptr() as usize;
                        vortex_ensure!(
                            address.is_multiple_of(direct_io_constraints.memory_alignment),
                            "host buffer address {address:#x} is not aligned to {} bytes",
                            direct_io_constraints.memory_alignment
                        );
                        let initialized = read_direct_at(
                            &file,
                            buffer.as_mut_slice(),
                            direct_range.read_offset,
                            direct_range.requested_range.end,
                            direct_io_constraints,
                        )?;
                        buffer.as_mut_slice()[initialized..].fill(0);
                        return Ok(BufferHandle::new_host(slice_direct_io_buffer(
                            buffer.freeze(),
                            direct_range.requested_range,
                            alignment,
                        )));
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

#[cfg(any(target_os = "linux", test))]
fn slice_direct_io_buffer(
    buffer: ByteBuffer,
    requested_range: std::ops::Range<usize>,
    alignment: Alignment,
) -> ByteBuffer {
    buffer.slice_unaligned(requested_range).aligned(alignment)
}

#[cfg(any(target_os = "linux", test))]
fn direct_io_range(offset: u64, length: usize, alignment: usize) -> VortexResult<DirectIoRange> {
    vortex_ensure!(alignment > 0, "direct I/O alignment must be non-zero");
    if length == 0 {
        return Ok(DirectIoRange {
            read_offset: offset,
            read_length: 0,
            requested_range: 0..0,
        });
    }

    let alignment_u64 = u64::try_from(alignment)?;
    let length_u64 = u64::try_from(length)?;
    let requested_end = offset.checked_add(length_u64).ok_or_else(|| {
        vortex_err!("direct I/O range overflow: offset={offset}, length={length}")
    })?;
    let read_offset = offset - offset % alignment_u64;
    let read_end = requested_end
        .checked_next_multiple_of(alignment_u64)
        .ok_or_else(|| vortex_err!("direct I/O aligned end overflow"))?;
    let read_length = usize::try_from(read_end - read_offset)?;
    let slice_start = usize::try_from(offset - read_offset)?;
    let slice_end = slice_start.checked_add(length).ok_or_else(|| {
        vortex_err!("direct I/O range overflow: offset={offset}, length={length}")
    })?;

    Ok(DirectIoRange {
        read_offset,
        read_length,
        requested_range: slice_start..slice_end,
    })
}

#[cfg(target_os = "linux")]
fn read_direct_at(
    file: &File,
    buffer: &mut [u8],
    offset: u64,
    required_bytes: usize,
    constraints: DirectIoConstraints,
) -> io::Result<usize> {
    let mut initialized = 0;
    while initialized < required_bytes {
        let initialized_u64 = u64::try_from(initialized)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "read offset overflow"))?;
        let read_offset = offset
            .checked_add(initialized_u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "read offset overflow"))?;
        let bytes_read = match file.read_at(&mut buffer[initialized..], read_offset) {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if bytes_read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "direct read returned {initialized} bytes, but {required_bytes} bytes were required"
                ),
            ));
        }
        initialized += bytes_read;
        if initialized < required_bytes
            && (!initialized.is_multiple_of(constraints.offset_alignment)
                || !initialized.is_multiple_of(constraints.memory_alignment))
        {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "direct read returned an unaligned short read of {initialized} bytes before the required {required_bytes} bytes"
                ),
            ));
        }
    }

    Ok(initialized)
}

#[cfg(target_os = "linux")]
fn direct_io_constraints(file: &File) -> VortexResult<DirectIoConstraints> {
    let Ok(stat) = rustix::fs::statx(
        file,
        c"",
        AtFlags::EMPTY_PATH | AtFlags::STATX_DONT_SYNC,
        StatxFlags::DIOALIGN,
    ) else {
        return Ok(DirectIoConstraints::default());
    };
    if stat.stx_mask & StatxFlags::DIOALIGN.bits() == 0 {
        return Ok(DirectIoConstraints::default());
    }

    let Ok(memory_alignment) = usize::try_from(stat.stx_dio_mem_align) else {
        return Ok(DirectIoConstraints::default());
    };
    let Ok(offset_alignment) = usize::try_from(stat.stx_dio_offset_align) else {
        return Ok(DirectIoConstraints::default());
    };
    if memory_alignment == 0 || offset_alignment == 0 {
        return Ok(DirectIoConstraints::default());
    }
    vortex_ensure!(
        memory_alignment.is_power_of_two(),
        "direct I/O memory alignment must be a power of two, got {memory_alignment}"
    );
    vortex_ensure!(
        offset_alignment.is_power_of_two(),
        "direct I/O offset alignment must be a power of two, got {offset_alignment}"
    );

    Ok(DirectIoConstraints {
        memory_alignment,
        offset_alignment,
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(0, 0, 4096, 0, 0, 0)]
    #[case(5, 0, 4096, 5, 0, 0)]
    #[case(5, 10, 4096, 0, 4096, 5)]
    #[case(4090, 20, 4096, 0, 8192, 4090)]
    #[case(4096, 4096, 4096, 4096, 4096, 0)]
    #[case(513, 1, 512, 512, 512, 1)]
    #[case(4096, 8193, 4096, 4096, 12288, 0)]
    fn widens_direct_read_to_block_boundaries(
        #[case] offset: u64,
        #[case] length: usize,
        #[case] alignment: usize,
        #[case] expected_offset: u64,
        #[case] expected_length: usize,
        #[case] expected_prefix: usize,
    ) -> VortexResult<()> {
        assert_eq!(
            direct_io_range(offset, length, alignment)?,
            DirectIoRange {
                read_offset: expected_offset,
                read_length: expected_length,
                requested_range: expected_prefix..expected_prefix + length,
            }
        );
        Ok(())
    }

    #[rstest]
    #[case(u64::MAX, 2, 4096)]
    #[case(0, 1, 0)]
    fn rejects_invalid_direct_read_range(
        #[case] offset: u64,
        #[case] length: usize,
        #[case] alignment: usize,
    ) {
        assert!(direct_io_range(offset, length, alignment).is_err());
    }

    #[test]
    fn aligned_ranges_cover_requested_bytes() -> VortexResult<()> {
        for alignment in [512, 4096] {
            for offset in 0..alignment * 2 {
                for length in [0, 1, alignment - 1, alignment, alignment + 1] {
                    let range = direct_io_range(offset as u64, length, alignment)?;
                    if length == 0 {
                        assert_eq!(range.read_length, 0);
                        continue;
                    }

                    assert_eq!(range.read_offset % alignment as u64, 0);
                    assert_eq!(range.read_length % alignment, 0);
                    assert_eq!(range.requested_range.len(), length);
                    assert!(range.requested_range.end <= range.read_length);
                    assert_eq!(
                        range.read_offset + range.requested_range.start as u64,
                        offset as u64
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn realigns_an_unaligned_direct_read_slice() -> VortexResult<()> {
        let range = direct_io_range(352, 1, 512)?;
        let requested_alignment = Alignment::new(512);
        let buffer = slice_direct_io_buffer(
            ByteBuffer::zeroed_aligned(range.read_length, requested_alignment),
            range.requested_range,
            requested_alignment,
        );

        assert_eq!(buffer.len(), 1);
        assert!(buffer.is_aligned(requested_alignment));
        Ok(())
    }
}
