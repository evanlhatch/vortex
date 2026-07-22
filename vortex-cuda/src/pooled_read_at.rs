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
    direct_io: bool,
    #[cfg(target_os = "linux")]
    direct_io_alignment: DirectIoAlignment,
}

impl PooledFileReadAt {
    /// Open a file for pooled reading with direct device transfer.
    pub fn open(
        path: impl AsRef<Path>,
        handle: Handle,
        pool: Arc<PinnedByteBufferPool>,
        stream: VortexCudaStream,
    ) -> VortexResult<Self> {
        let path = path.as_ref();
        let uri = Arc::from(path.to_string_lossy().to_string());
        let file = Arc::new(File::open(path)?);
        Ok(Self {
            uri,
            file,
            handle,
            pool,
            stream,
            direct_io: false,
            #[cfg(target_os = "linux")]
            direct_io_alignment: DirectIoAlignment::default(),
        })
    }

    /// Open a file for pooled direct I/O with direct device transfer.
    ///
    /// Direct I/O bypasses the operating system page cache. Unaligned logical reads are widened
    /// to aligned physical reads and sliced to the requested range after transfer to the device.
    #[cfg(target_os = "linux")]
    pub fn open_direct(
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
        let direct_io_alignment = direct_io_alignment(&file);
        let file = Arc::new(file);
        Ok(Self {
            uri,
            file,
            handle,
            pool,
            stream,
            direct_io: true,
            direct_io_alignment,
        })
    }

    /// Return an error when direct I/O is requested on an unsupported platform.
    #[cfg(not(target_os = "linux"))]
    pub fn open_direct(
        _path: impl AsRef<Path>,
        _handle: Handle,
        _pool: Arc<PinnedByteBufferPool>,
        _stream: VortexCudaStream,
    ) -> VortexResult<Self> {
        vortex::error::vortex_bail!("direct CUDA file I/O is only supported on Linux")
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
        let direct_io = self.direct_io;
        #[cfg(target_os = "linux")]
        let direct_io_alignment = self.direct_io_alignment;

        async move {
            #[cfg(target_os = "linux")]
            let (read_offset, read_length, requested_range) = if direct_io {
                direct_io_range(offset, length, direct_io_alignment.offset)?
            } else {
                (offset, length, 0..length)
            };
            #[cfg(not(target_os = "linux"))]
            let (read_offset, read_length, requested_range) = {
                vortex_ensure!(
                    !direct_io,
                    "direct CUDA file I/O is only supported on Linux"
                );
                (offset, length, 0..length)
            };
            #[cfg(target_os = "linux")]
            let required_bytes = requested_range.end;

            let mut target = pool.get(read_length)?;
            let target = handle
                .spawn_blocking(move || {
                    #[cfg(target_os = "linux")]
                    if direct_io {
                        let address = target.as_mut_slice().as_ptr() as usize;
                        if !address.is_multiple_of(direct_io_alignment.memory) {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidInput,
                                format!(
                                    "pinned buffer address {address:#x} is not aligned to {} bytes",
                                    direct_io_alignment.memory
                                ),
                            ));
                        }

                        let bytes_read = read_direct_at(
                            &file,
                            target.as_mut_slice(),
                            read_offset,
                            required_bytes,
                            direct_io_alignment,
                        )?;
                        target.truncate(bytes_read);
                    } else {
                        read_exact_at(&file, target.as_mut_slice(), read_offset)?;
                    }
                    #[cfg(not(target_os = "linux"))]
                    read_exact_at(&file, target.as_mut_slice(), read_offset)?;
                    Ok::<_, io::Error>(target)
                })
                .await
                .map_err(VortexError::from)?;

            let cuda_buf = target.transfer_to_device(&stream)?;
            Ok(BufferHandle::new_device(Arc::new(cuda_buf)).slice(requested_range))
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
