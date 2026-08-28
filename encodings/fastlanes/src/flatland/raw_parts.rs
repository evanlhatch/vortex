// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! RawParts typed views for the fastlanes encodings (flatland REBUILD
//! Part 3 #3 — completes the "8 impls": FoR + BitPacked join Primitive/
//! Constant/Bool/Dict).
//!
//! These expose the *physical* representation the cursor decodes from —
//! encoded child + reference (FoR), packed words + bit width + offset +
//! patches (BitPacked) — NOT decoded values. `RawParts::raw_parts` returns
//! `None` on encoding/ptype mismatch; never allocates, never canonicalizes.
//!
//! Parts are `Copy`: children are borrowed `&ArrayRef` via the array's slot
//! list (inherent accessor, no view-lifetime trap); scalars are copied as
//! [`PValue`]; the packed payload is a raw `(*const u8, len)` view. SAFETY:
//! the raw pointer is valid for the array's lifetime — cursor owners must
//! uphold the engine's pin-free (ptr, epoch) discipline (REBUILD Part 12:
//! base arrays are immutable within a tick).

use std::marker::PhantomData;

use vortex_array::ArrayRef;
use vortex_array::dtype::{DType, NativePType};
use vortex_array::flatland::raw_parts::RawParts;
use vortex_array::scalar::PValue;

/// Physical view of a frame-of-reference column: the reference the encoded
/// values are offsets from, plus the encoded child (a primitive or bitpacked
/// array of deltas). Decoding = `encoded[row] + reference`.
#[derive(Debug, Clone, Copy)]
pub struct FoRParts<'a> {
    /// The frame reference (subtracted during encode), as a typed value.
    pub reference: PValue,
    /// The delta-encoded child (Primitive or BitPacked).
    pub encoded: &'a ArrayRef,
}

unsafe impl<T: NativePType> RawParts<T> for crate::FoR {
    type Parts<'a> = FoRParts<'a>;
    fn raw_parts<'a>(array: &'a ArrayRef) -> Option<Self::Parts<'a>> {
        use crate::r#for::FoRArrayExt as _;
        if !ptype_matches::<T>(array.dtype()) {
            return None;
        }
        let encoded = array
            .slots()
            .get(crate::FoRSlots::ENCODED)
            .and_then(|s| s.as_ref())?;
        // pvalue() is Copy — extracted, never escaped from the local view.
        let reference = array
            .as_opt::<crate::FoR>()?
            .reference_scalar()
            .as_primitive()
            .pvalue()?;
        Some(FoRParts { reference, encoded })
    }
}

/// Physical view of a bit-packed column: packed words, bit width, and
/// offset. Decoding = unpack words at `bit_width`, then apply exception
/// patches (fetched via `BitPackedArrayExt::patches` at decode time —
/// `Patches` is not `Copy`, so it can't ride in a `Copy` part).
#[derive(Debug, Clone, Copy)]
pub struct BitPackedParts<'a> {
    /// Packed words (host-resident bytes; valid for the array's lifetime).
    pub packed: *const u8,
    /// Packed byte length.
    pub packed_len: usize,
    /// Bits per value.
    pub bit_width: u8,
    /// Bit offset into the first byte.
    pub offset: u16,
    /// Structural lifetime (packed points into the array's buffer).
    _marker: PhantomData<&'a [u8]>,
}

unsafe impl<T: NativePType> RawParts<T> for crate::BitPacked {
    type Parts<'a> = BitPackedParts<'a>;
    fn raw_parts<'a>(array: &'a ArrayRef) -> Option<Self::Parts<'a>> {
        use crate::bitpacking::BitPackedArrayExt as _;
        if !ptype_matches::<T>(array.dtype()) {
            return None;
        }
        let view = array.as_opt::<crate::BitPacked>()?;
        let host = view.packed().as_host_opt()?;
        let bytes = host.as_slice();
        // All extracted pieces are Copy/owned; nothing is borrowed past this
        // statement (the packed pointer stays valid via the array lifetime).
        Some(BitPackedParts {
            packed: bytes.as_ptr(),
            packed_len: bytes.len(),
            bit_width: view.bit_width(),
            offset: view.offset(),
            _marker: PhantomData,
        })
    }
}

/// Does `T`'s ptype match this dtype (nullability-agnostic)?
fn ptype_matches<T: NativePType>(dtype: &DType) -> bool {
    matches!(dtype, DType::Primitive(pt, _) if *pt == T::PTYPE)
}
