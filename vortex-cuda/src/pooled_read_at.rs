// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io;
#[cfg(target_os = "linux")]
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

use futures::FutureExt;
use futures::StreamExt;
use futures::future::BoxFuture;
use object_store::GetOptions;
use object_store::GetRange;
use object_store::GetResultPayload;
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectPath;
#[cfg(target_os = "linux")]
use rustix::fs::AtFlags;
#[cfg(target_os = "linux")]
use rustix::fs::Mode;
#[cfg(target_os = "linux")]
use rustix::fs::OFlags;
#[cfg(target_os = "linux")]
use rustix::fs::StatxFlags;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBuffer;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::io::CoalesceConfig;
use vortex::io::VortexReadAt;
use vortex::io::runtime::Handle;
use vortex::io::std_file::read_exact_at;

use crate::pinned::PinnedByteBufferPool;
use crate::stream::VortexCudaStream;

/// Default number of concurrent requests to allow for local file I/O.
pub const DEFAULT_FILE_CONCURRENCY: usize = 32;
/// Default number of concurrent requests to allow for object store I/O.
pub const DEFAULT_OBJECT_STORE_CONCURRENCY: usize = 192;

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

#[derive(Debug, PartialEq, Eq)]
struct DirectIoRange {
    read_offset: u64,
    read_length: usize,
    requested_range: std::ops::Range<usize>,
}

/// Options controlling how [`PooledFileReadAt`] opens and reads a local file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PooledFileReadAtOptions {
    direct_io: bool,
}

impl PooledFileReadAtOptions {
    /// Bypass the operating system page cache for pooled file reads.
    ///
    /// This option is available only on Linux. Unaligned logical reads are widened to satisfy the
    /// filesystem's direct-I/O requirements and sliced back to the requested range after transfer
    /// to the device.
    #[cfg(target_os = "linux")]
    pub fn with_direct_io(mut self) -> Self {
        self.direct_io = true;
        self
    }
}

/// File reader that uses CUDA pinned host memory for I/O buffers and transfers
/// directly to the GPU.
///
/// Reads into a pooled pinned (page-locked) buffer, then submits a non-blocking
/// H2D DMA transfer and returns a device `BufferHandle`.
///
/// This is a data-plane reader. To open a complete local Vortex file, prefer
/// [`crate::CudaOpenOptionsExt::with_cuda`], which keeps the footer and zone maps on the host.
#[derive(Clone)]
pub struct PooledFileReadAt {
    uri: Arc<str>,
    file: Arc<File>,
    handle: Handle,
    pool: Arc<PinnedByteBufferPool>,
    stream: VortexCudaStream,
    #[cfg(target_os = "linux")]
    direct_io: bool,
    #[cfg(target_os = "linux")]
    direct_io_constraints: DirectIoConstraints,
}

impl PooledFileReadAt {
    /// Open a file for pooled reading with direct device transfer.
    pub fn open(
        path: impl AsRef<Path>,
        handle: Handle,
        pool: Arc<PinnedByteBufferPool>,
        stream: VortexCudaStream,
    ) -> VortexResult<Self> {
        Self::open_with_options(
            path,
            handle,
            pool,
            stream,
            PooledFileReadAtOptions::default(),
        )
    }

    /// Open a file for pooled reading with explicit options.
    pub fn open_with_options(
        path: impl AsRef<Path>,
        handle: Handle,
        pool: Arc<PinnedByteBufferPool>,
        stream: VortexCudaStream,
        options: PooledFileReadAtOptions,
    ) -> VortexResult<Self> {
        #[cfg(target_os = "linux")]
        if options.direct_io {
            return Self::open_direct(path, handle, pool, stream);
        }
        #[cfg(not(target_os = "linux"))]
        let _ = options;

        let path = path.as_ref();
        let uri = Arc::from(path.to_string_lossy().to_string());
        let file = Arc::new(File::open(path)?);
        Ok(Self {
            uri,
            file,
            handle,
            pool,
            stream,
            #[cfg(target_os = "linux")]
            direct_io: false,
            #[cfg(target_os = "linux")]
            direct_io_constraints: DirectIoConstraints::default(),
        })
    }

    #[cfg(target_os = "linux")]
    fn open_direct(
        path: impl AsRef<Path>,
        handle: Handle,
        pool: Arc<PinnedByteBufferPool>,
        stream: VortexCudaStream,
    ) -> VortexResult<Self> {
        let path = path.as_ref();
        let uri = Arc::from(path.to_string_lossy().to_string());
        let file = File::from(
            rustix::fs::open(
                path,
                OFlags::RDONLY | OFlags::CLOEXEC | OFlags::DIRECT,
                Mode::empty(),
            )
            .map_err(io::Error::from)?,
        );
        let direct_io_constraints = direct_io_constraints(&file)?;
        let file = Arc::new(file);
        Ok(Self {
            uri,
            file,
            handle,
            pool,
            stream,
            direct_io: true,
            direct_io_constraints,
        })
    }
}

impl VortexReadAt for PooledFileReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(CoalesceConfig::file())
    }

    fn concurrency(&self) -> usize {
        DEFAULT_FILE_CONCURRENCY
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
        _alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let file = Arc::clone(&self.file);
        let handle = self.handle.clone();
        let stream = self.stream.clone();
        let pool = Arc::clone(&self.pool);
        #[cfg(target_os = "linux")]
        let direct_io = self.direct_io;
        #[cfg(target_os = "linux")]
        let direct_io_constraints = self.direct_io_constraints;

        async move {
            #[cfg(target_os = "linux")]
            let direct_range = if direct_io {
                direct_io_range(offset, length, direct_io_constraints.offset_alignment)?
            } else {
                DirectIoRange {
                    read_offset: offset,
                    read_length: length,
                    requested_range: 0..length,
                }
            };
            #[cfg(not(target_os = "linux"))]
            let direct_range = DirectIoRange {
                read_offset: offset,
                read_length: length,
                requested_range: 0..length,
            };

            let mut target = pool.get(direct_range.read_length)?;
            let target = handle
                .spawn_blocking(move || {
                    #[cfg(target_os = "linux")]
                    if direct_io {
                        let address = target.as_mut_slice().as_ptr() as usize;
                        if !address.is_multiple_of(direct_io_constraints.memory_alignment) {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidInput,
                                format!(
                                    "pinned buffer address {address:#x} is not aligned to {} bytes",
                                    direct_io_constraints.memory_alignment
                                ),
                            ));
                        }

                        let bytes_read = read_direct_at(
                            &file,
                            target.as_mut_slice(),
                            direct_range.read_offset,
                            direct_range.requested_range.end,
                            direct_io_constraints,
                        )?;
                        target.truncate(bytes_read);
                    } else {
                        read_exact_at(&file, target.as_mut_slice(), direct_range.read_offset)?;
                    }
                    #[cfg(not(target_os = "linux"))]
                    read_exact_at(&file, target.as_mut_slice(), direct_range.read_offset)?;
                    Ok::<_, io::Error>(target)
                })
                .await
                .map_err(VortexError::from)?;

            let cuda_buf = target.transfer_to_device(&stream)?;
            Ok(BufferHandle::new_device(Arc::new(cuda_buf)).slice(direct_range.requested_range))
        }
        .boxed()
    }
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

/// Object store reader that uses CUDA pinned host memory for I/O buffers and
/// transfers directly to the GPU.
///
/// Reads into a pooled pinned (page-locked) buffer, then submits a non-blocking
/// H2D DMA transfer and returns a device `BufferHandle`.
#[derive(Clone)]
pub struct PooledObjectStoreReadAt {
    store: Arc<dyn ObjectStore>,
    path: ObjectPath,
    uri: Arc<str>,
    handle: Handle,
    pool: Arc<PinnedByteBufferPool>,
    stream: VortexCudaStream,
    concurrency: usize,
    coalesce_config: Option<CoalesceConfig>,
}

impl PooledObjectStoreReadAt {
    /// Create a new object-store source with pinned host-buffer allocations and direct device transfer.
    pub fn new(
        store: Arc<dyn ObjectStore>,
        path: ObjectPath,
        handle: Handle,
        pool: Arc<PinnedByteBufferPool>,
        stream: VortexCudaStream,
    ) -> Self {
        let uri = Arc::from(path.to_string());
        Self {
            store,
            path,
            uri,
            handle,
            pool,
            stream,
            concurrency: DEFAULT_OBJECT_STORE_CONCURRENCY,
            coalesce_config: Some(CoalesceConfig::object_storage()),
        }
    }

    /// Set the concurrency for this source.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Set the coalesce config for this source.
    pub fn with_coalesce_config(mut self, config: CoalesceConfig) -> Self {
        self.coalesce_config = Some(config);
        self
    }
}

impl VortexReadAt for PooledObjectStoreReadAt {
    fn uri(&self) -> Option<&Arc<str>> {
        Some(&self.uri)
    }

    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        self.coalesce_config
    }

    fn concurrency(&self) -> usize {
        self.concurrency
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let store = Arc::clone(&self.store);
        let path = self.path.clone();
        async move {
            store
                .head(&path)
                .await
                .map(|h| h.size)
                .map_err(VortexError::from)
        }
        .boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        _alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let store = Arc::clone(&self.store);
        let path = self.path.clone();
        let handle = self.handle.clone();
        let stream = self.stream.clone();
        let pool = Arc::clone(&self.pool);

        async move {
            let end = offset.checked_add(length as u64).ok_or_else(|| {
                vortex_err!(
                    "Object store read range overflow: offset={}, length={}",
                    offset,
                    length
                )
            })?;
            let range = offset..end;
            let mut target = pool.get(length)?;
            let response = store
                .get_opts(
                    &path,
                    GetOptions {
                        range: Some(GetRange::Bounded(range.clone())),
                        ..Default::default()
                    },
                )
                .await?;

            match response.payload {
                #[cfg(not(target_arch = "wasm32"))]
                GetResultPayload::File(file, _) => {
                    target = handle
                        .spawn_blocking(move || {
                            read_exact_at(&file, target.as_mut_slice(), range.start)?;
                            Ok::<_, io::Error>(target)
                        })
                        .await
                        .map_err(VortexError::from)?;
                }
                #[cfg(target_arch = "wasm32")]
                GetResultPayload::File(..) => {
                    unreachable!("File payload not supported on wasm32")
                }
                GetResultPayload::Stream(mut byte_stream) => {
                    let mut filled = 0usize;
                    while let Some(bytes) = byte_stream.next().await {
                        let bytes = bytes?;
                        let end = filled + bytes.len();
                        vortex_ensure!(
                            end <= length,
                            "Object store stream returned more bytes than expected (expected {} bytes, got at least {} bytes, range: {:?})",
                            length,
                            end,
                            range
                        );
                        target.as_mut_slice()[filled..end].copy_from_slice(&bytes);
                        filled = end;
                    }

                    vortex_ensure!(
                        filled == length,
                        "Object store stream returned {} bytes but expected {} bytes (range: {:?})",
                        filled,
                        length,
                        range
                    );
                }
            }

            let cuda_buf = target.transfer_to_device(&stream)?;
            Ok(BufferHandle::new_device(Arc::new(cuda_buf)))
        }
        .boxed()
    }
}

/// Default number of concurrent requests to allow for in-memory byte buffer I/O.
pub const DEFAULT_BYTE_BUFFER_CONCURRENCY: usize = 16;

/// In-memory byte buffer reader that uses CUDA pinned host memory for staging
/// and transfers directly to the GPU.
///
/// Slices the source `ByteBuffer`, copies into a pooled pinned (page-locked)
/// buffer, then submits a non-blocking H2D DMA transfer and returns a device
/// `BufferHandle`.
#[derive(Clone)]
pub struct PooledByteBufferReadAt {
    buffer: ByteBuffer,
    pool: Arc<PinnedByteBufferPool>,
    stream: VortexCudaStream,
}

impl PooledByteBufferReadAt {
    /// Create a new in-memory reader with pinned host-buffer allocations and direct device transfer.
    pub fn new(
        buffer: ByteBuffer,
        pool: Arc<PinnedByteBufferPool>,
        stream: VortexCudaStream,
    ) -> Self {
        Self {
            buffer,
            pool,
            stream,
        }
    }
}

impl VortexReadAt for PooledByteBufferReadAt {
    fn coalesce_config(&self) -> Option<CoalesceConfig> {
        Some(CoalesceConfig::in_memory())
    }

    fn concurrency(&self) -> usize {
        DEFAULT_BYTE_BUFFER_CONCURRENCY
    }

    fn size(&self) -> BoxFuture<'static, VortexResult<u64>> {
        let len = self.buffer.len() as u64;
        async move { Ok(len) }.boxed()
    }

    fn read_at(
        &self,
        offset: u64,
        length: usize,
        _alignment: Alignment,
    ) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        let buffer = self.buffer.clone();
        let stream = self.stream.clone();
        let pool = Arc::clone(&self.pool);

        async move {
            let offset = usize::try_from(offset)
                .map_err(|_| vortex_err!("Byte buffer read offset overflow: offset={}", offset))?;
            let src = &buffer.as_ref()[offset..offset + length];

            let mut target = pool.get(length)?;
            target.as_mut_slice().copy_from_slice(src);

            let cuda_buf = target.transfer_to_device(&stream)?;
            Ok(BufferHandle::new_device(Arc::new(cuda_buf)))
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn pooled_file_read_options_default_to_buffered_io() {
        assert!(!PooledFileReadAtOptions::default().direct_io);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn pooled_file_read_options_enable_direct_io() {
        assert!(
            PooledFileReadAtOptions::default()
                .with_direct_io()
                .direct_io
        );
    }

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
}
