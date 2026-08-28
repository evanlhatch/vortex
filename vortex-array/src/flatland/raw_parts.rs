// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Typed, zero-copy, borrowed views over an encoding's physical data
//! (flatland REBUILD Part 3 #3).
//!
//! Read-side analogue of [`crate::ArrayRef::try_buffer_mut`]: a cursor can
//! obtain a `Copy` view of an encoding's physical representation without
//! canonicalizing. `raw_parts` returns `None` when the array's encoding or
//! element ptype does not match; host-residency is assumed for the encodings
//! backed by contiguous byte buffers (device arrays are outside the engine
//! cursor's scope).
//!
//! SAFETY contract of implementations: the returned parts borrow from the
//! input array (`'a`), so they are valid exactly as long as the array;
//! cursor owners must uphold the engine's pin-free (ptr, epoch) discipline
//! (flatland REBUILD Part 12: base arrays are immutable within a tick).

use vortex_buffer::BitBufferView;

use crate::ArrayRef;
use crate::VTable;
use crate::array::ArrayData;
use crate::array::DynArrayData;
use crate::arrays::Bool;
use crate::arrays::Constant;
use crate::arrays::Dict;
use crate::arrays::dict::DictSlots;
use crate::arrays::chunked::ChunkedSlots;
use crate::arrays::Primitive;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;

/// Typed raw view provider. `T` is the encoded element type (the logical
/// dtype's primitive ptype), checked against the array's actual dtype.
///
/// # Safety
/// Implementations must return parts borrowed from the input array and must
/// not hand out aliased mutable state. `Parts` must be `Copy` (raw pointers
/// and references only — no owned allocations).
pub unsafe trait RawParts<T: NativePType>: VTable {
    /// Zero-copy borrowed view over the encoding's physical data.
    type Parts<'a>: Copy;
    /// Borrow the physical data, or `None` if encoding/ptype mismatch.
    fn raw_parts<'a>(array: &'a ArrayRef) -> Option<Self::Parts<'a>>;
}

/// Row-decode accessor: parts that can produce a typed value for row `i` in
/// O(1) (no loop, no allocation). This is the cursor's per-row read — the
/// engine never writes a per-encoding decode loop.
pub trait DecodeRow<T> {
    /// Typed value at row `i`. `None` when the part cannot decode `T` or the
    /// row is out of range.
    fn value(&self, i: usize) -> Option<T>;
}

impl<'a, T: NativePType> DecodeRow<T> for PrimitiveParts<'a, T> {
    #[inline]
    fn value(&self, i: usize) -> Option<T> {
        self.values.get(i).copied()
    }
}

impl<'a, T: NativePType> DecodeRow<T> for ConstantParts<'a> {
    #[inline]
    fn value(&self, i: usize) -> Option<T> {
        if i >= self.len {
            return None;
        }
        let pv = self.scalar.as_primitive_opt()?.pvalue()?;
        pv.cast::<T>().ok()
    }
}

/// Borrow the encoding-specific data for `V` with the array's lifetime.
fn typed_data<'a, V: VTable>(array: &'a ArrayRef) -> Option<&'a V::TypedArrayData> {
    let dyn_data: &'a dyn DynArrayData = array.dyn_array();
    Some(&dyn_data.as_any().downcast_ref::<ArrayData<V>>()?.data)
}

/// Physical view of a primitive column: the typed value slice.
#[derive(Debug, Clone, Copy)]
pub struct PrimitiveParts<'a, T> {
    /// Typed values (nonsense for null rows — check validity separately).
    pub values: &'a [T],
}

unsafe impl<T: NativePType> RawParts<T> for Primitive {
    type Parts<'a> = PrimitiveParts<'a, T>;
    fn raw_parts<'a>(array: &'a ArrayRef) -> Option<Self::Parts<'a>> {
        let data = typed_data::<Primitive>(array)?;
        if data.ptype() != T::PTYPE {
            return None;
        }
        if data.buffer_handle().as_host_opt().is_none() {
            return None;
        }
        Some(PrimitiveParts {
            values: data.as_slice::<T>(),
        })
    }
}

/// Physical view of a constant column: the scalar plus its length.
#[derive(Debug, Clone, Copy)]
pub struct ConstantParts<'a> {
    /// The scalar repeated across the column.
    pub scalar: &'a Scalar,
    /// Logical column length.
    pub len: usize,
}

unsafe impl<T: NativePType> RawParts<T> for Constant {
    type Parts<'a> = ConstantParts<'a>;
    fn raw_parts<'a>(array: &'a ArrayRef) -> Option<Self::Parts<'a>> {
        let data = typed_data::<Constant>(array)?;
        if data.scalar().dtype() != &T::dtype_of() {
            return None;
        }
        Some(ConstantParts {
            scalar: data.scalar(),
            len: array.len(),
        })
    }
}

/// Physical view of a boolean column: packed bits (MSB convention per
/// [`BitBufferView`]).
#[derive(Debug, Clone, Copy)]
pub struct BoolParts<'a> {
    /// Packed bits.
    pub bits: BitBufferView<'a>,
}

unsafe impl RawParts<u8> for Bool {
    type Parts<'a> = BoolParts<'a>;
    fn raw_parts<'a>(array: &'a ArrayRef) -> Option<Self::Parts<'a>> {
        let data = typed_data::<Bool>(array)?;
        Some(BoolParts {
            bits: data.bit_buffer_view(),
        })
    }
}

/// Physical view of a dictionary column: codes and values as child arrays.
/// Codes may be any unsigned ptype (the cursor matches); values carry the
/// logical column dtype.
#[derive(Debug, Clone, Copy)]
pub struct DictParts<'a> {
    /// Codes array (unsigned integer, nullable validity).
    pub codes: &'a ArrayRef,
    /// Distinct values.
    pub values: &'a ArrayRef,
    /// Logical column length.
    pub len: usize,
}

unsafe impl<T: NativePType> RawParts<T> for Dict {
    type Parts<'a> = DictParts<'a>;
    fn raw_parts<'a>(array: &'a ArrayRef) -> Option<Self::Parts<'a>> {
        if array.dtype() != &T::dtype_of() {
            return None;
        }
        let codes = array
            .slots()
            .get(DictSlots::CODES)
            .and_then(|s| s.as_ref())?;
        let values = array
            .slots()
            .get(DictSlots::VALUES)
            .and_then(|s| s.as_ref())?;
        if values.dtype() != &<T as NativeDType>::dtype_of() {
            return None;
        }
        Some(DictParts {
            codes,
            values,
            len: array.len(),
        })
    }
}

impl<T: NativePType> NativeDType for T {
    fn dtype_of() -> DType {
        DType::Primitive(T::PTYPE, Nullability::NonNullable)
    }
}

pub(crate) trait NativeDType {
    fn dtype_of() -> DType;
}

/// Physical view of a chunked column: the chunk list the cursor walks
/// (REBUILD Part 13: spawn = Chunked lazy append, the read cursor walks
/// chunks, base chunk hot). Offsets give the row range of each chunk
/// (cumulative u64; `chunks[i]` covers rows `[offsets[i]..offsets[i+1])`).
#[derive(Debug, Clone, Copy)]
pub struct ChunkedParts<'a> {
    /// Children in slot order; `chunks[i]` is the i-th chunk, all sharing
    /// the outer dtype.
    pub chunks: &'a [Option<ArrayRef>],
    /// Cumulative row offsets (u64), length nchunks+1.
    pub chunk_offsets: &'a ArrayRef,
    /// Logical row count.
    pub len: usize,
}

unsafe impl<T: NativePType> RawParts<T> for crate::arrays::Chunked {
    type Parts<'a> = ChunkedParts<'a>;
    fn raw_parts<'a>(array: &'a ArrayRef) -> Option<Self::Parts<'a>> {
        use crate::arrays::chunked::ChunkedArrayExt as _;
        if array.dtype() != &T::dtype_of() {
            return None;
        }
        let slots = array.slots();
        let chunk_offsets = slots
            .get(ChunkedSlots::CHUNK_OFFSETS)
            .and_then(|s| s.as_ref())?;
        // Variadic tail: children are slots [CHUNK_OFFSETS+1..].
        let first_chunk = ChunkedSlots::CHUNK_OFFSETS + 1;
        let chunks = &slots[first_chunk..];
        Some(ChunkedParts {
            chunks,
            chunk_offsets,
            len: array.len(),
        })
    }
}
