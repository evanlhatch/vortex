// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stable ascending sort over array rows, returning the row permutation.
//!
//! [`sort_to_indices`] returns a non-nullable `u32` [`PrimitiveArray`] of row indices such that
//! taking the input array at those indices yields its elements in sorted (ascending) order.
//! The sort is stable: equal elements retain their original relative order.
//!
//! **Null ordering:** following the fork's `is_sorted` semantics, nulls sort *first* (a
//! non-null element before a null element violates the `Stat::IsSorted` order).
//!
//! Per-encoding fast paths (see [`sort_to_indices`]):
//! - `Constant`: every element is equal, so the identity permutation `[0..n)` is sorted.
//! - `Primitive`: if the cached `Stat::IsSorted` is `exact(true)`, the array is already sorted and
//!   the identity permutation is returned. Otherwise a stable radix sort for `u32`/`u64` keys, and
//!   a stable `sort_by` over the native ordering for all other primitive types.
//! - `Dict`: the dictionary *values* are sorted once (O(dict log dict)), then every code is
//!   remapped to the sorted-rank of its value; the output row permutation is a stable sort of the
//!   remapped codes. This avoids sorting O(N) row *values*.
//! - Fallback (any other encoding): materialize [`Canonical`] once and sort the canonical form.
//!   Non-numeric canonical forms (strings, lists, structs, …) are not yet supported and return an
//!   explicit `NotImplemented` error rather than producing wrong results.

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Constant;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProviderExt;
use crate::match_each_integer_ptype;
use crate::match_each_native_ptype;

/// Returns a stable ascending sort of `arr` as a non-nullable `u32` array of row indices.
///
/// Taking `arr` at the returned indices yields the elements of `arr` in sorted order; the sort is
/// stable and places nulls first. See the [module docs](self) for the per-encoding fast paths.
///
/// Non-numeric dtypes (strings, lists, structs, …) are not yet supported and return an
/// [`crate::VortexError`] of kind `NotImplemented`.
pub fn sort_to_indices(arr: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let n = arr.len();

    // Arrays of length 0 or 1 are always sorted: the identity permutation is a valid stable sort.
    if n <= 1 {
        return Ok(identity_indices(n));
    }

    // Cached IsSorted stat: if the array is proven already-sorted, the identity permutation is the
    // stable sort (equal keys keep their original order). We only trust an exact cached result; an
    // absent or inexact stat falls through to a full sort.
    if arr
        .statistics()
        .get_as::<bool>(Stat::IsSorted)
        .as_exact()
        == Some(true)
    {
        return Ok(identity_indices(n));
    }

    // Every constant array is sorted (all elements equal): identity permutation.
    if arr.is::<Constant>() {
        return Ok(identity_indices(n));
    }

    // Dict fast path: sort the dictionary values once and remap codes.
    if let Some(dict) = arr.as_opt::<Dict>() {
        return sort_dict_to_indices(dict.into_owned(), ctx);
    }

    // Primitive fast path: radix for u32/u64, stable `sort_by` otherwise.
    if arr.is::<Primitive>() {
        let primitive = arr.clone().execute::<PrimitiveArray>(ctx)?;
        return sort_primitive_to_indices(&primitive, ctx);
    }

    // Fallback: materialize the canonical form once and sort that. Non-primitive canonical forms
    // do not have a scalar-ordering implementation yet.
    let canonical = arr.clone().execute::<Canonical>(ctx)?;
    match canonical {
        Canonical::Primitive(primitive) => sort_primitive_to_indices(&primitive, ctx),
        _ => vortex_bail!(
            NotImplemented: "sort_to_indices",
            format!("vortex-array sort for non-primitive dtype {}", arr.dtype())
        ),
    }
}

/// The identity permutation `[0, 1, …, n)` as a non-nullable `u32` [`PrimitiveArray`].
fn identity_indices(n: usize) -> ArrayRef {
    // Sort indices are u32 by convention (flatland: ≤ 2^32 rows per column).
    #[allow(clippy::cast_possible_truncation, reason = "sort indices are u32 by convention; n is a column length")]
    PrimitiveArray::from_iter((0..n).map(|i| i as u32)).into_array()
}

/// Instantiate an array of `u32` row indices from a `Vec<u32>`.
fn indices_array(indices: Vec<u32>) -> ArrayRef {
    PrimitiveArray::from_iter(indices).into_array()
}

/// Stable LSD radix sort over `u64` keys, returning positions in ascending order.
///
/// `u32` keys are handled by [`radix_sort_indices_u32`], which widens once and reuses this core.
fn radix_sort_indices_u64(values: &[u64]) -> Vec<u32> {
    radix_sort_indices_core(values, 64)
}

/// Stable LSD radix sort over `u32` keys, returning positions in ascending order.
fn radix_sort_indices_u32(values: &[u32]) -> Vec<u32> {
    radix_sort_indices_core(
        &values.iter().map(|&v| v as u64).collect::<Vec<_>>(),
        32,
    )
}

/// Shared LSD radix core over widened `u64` keys.
///
/// `bit_width` is the native key width (32 or 64) so leading zero bytes above the key width are
/// never scanned. The sort is stable: equal keys keep their original relative order.
fn radix_sort_indices_core(values: &[u64], bit_width: u32) -> Vec<u32> {
    let len = values.len();
    // Row indices are u32 by convention (flatland: ≤ 2^32 rows per column).
    #[allow(clippy::cast_possible_truncation, reason = "u32 row-index convention")]
    let mut order: Vec<u32> = (0..len as u32).collect();
    let mut tmp: Vec<u32> = vec![0; len];

    // Skip leading zero bytes: only iterate up to the highest byte that contains a set bit.
    // `top` is the bitwise-OR of all key values; for u32 keys widened to u64 the upper 32 bits are
    // always zero, so clamp the leading-zero count to the key width before subtracting.
    let mut top = 0u64;
    for &v in values {
        top |= v;
    }
    // Significant bytes over the WIDENED u64 key (the u32-widening path has
    // 32+ leading zeros always — clamp to the NATIVE width's byte count,
    // never subtract against u64 leading zeros directly).
    let significant = (64 - top.leading_zeros()).div_ceil(8).max(1);
    let native_bytes = bit_width / 8;
    let highest_byte = (significant.min(native_bytes) - 1) as usize;

    for byte in 0..=highest_byte {
        let shift = byte * 8;
        let mut counts = [0usize; 256];
        for &v in values {
            counts[((v >> shift) & 0xff) as usize] += 1;
        }
        // Convert counts to starting offsets.
        let mut total = 0;
        for c in counts.iter_mut() {
            let t = *c;
            *c = total;
            total += t;
        }
        for &idx in &order {
            let v = values[idx as usize];
            let digit = ((v >> shift) & 0xff) as usize;
            tmp[counts[digit]] = idx;
            counts[digit] += 1;
        }
        std::mem::swap(&mut order, &mut tmp);
    }
    order
}

/// Stable ascending sort of a primitive array, returning original row indices.
///
/// Nulls sort first (in their original relative order, preserving stability); non-null elements
/// are sorted ascending, with equal keys keeping their original relative order.
fn sort_primitive_to_indices(
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // Nulls first. Build the validity mask once.
    let mask = array.validity()?.execute_mask(array.len(), ctx)?;
    let mut null_indices: Vec<u32> = Vec::new();
    let mut valid_indices: Vec<u32> = Vec::new();
    // Row indices are u32 by convention (flatland: ≤ 2^32 rows per column).
    #[allow(clippy::cast_possible_truncation, reason = "u32 row-index convention")]
    for (idx, is_valid) in mask.iter().enumerate() {
        if is_valid {
            valid_indices.push(idx as u32);
        } else {
            null_indices.push(idx as u32);
        }
    }

    let mut order = null_indices;

    // Dispatch on ptype first so each arm is only type-checked for its concrete native type.
    // u32/u64 get the radix fast path; everything else gets a stable `sort_by` over the native
    // total ordering (`total_compare` handles floats, including NaN placement).
    match array.ptype() {
        PType::U32 => {
            let values = array.to_buffer::<u32>();
            let compacted: Vec<u32> = valid_indices
                .iter()
                .map(|&i| values[i as usize])
                .collect();
            order.extend(
                radix_sort_indices_u32(&compacted)
                    .into_iter()
                    .map(|pos| valid_indices[pos as usize]),
            );
        }
        PType::U64 => {
            let values = array.to_buffer::<u64>();
            let compacted: Vec<u64> = valid_indices
                .iter()
                .map(|&i| values[i as usize])
                .collect();
            order.extend(
                radix_sort_indices_u64(&compacted)
                    .into_iter()
                    .map(|pos| valid_indices[pos as usize]),
            );
        }
        _ => match_each_native_ptype!(array.ptype(), |T| {
            let values = array.to_buffer::<T>();
            let mut sorted = valid_indices;
            sorted.sort_by(|&a, &b| values[a as usize].total_compare(values[b as usize]));
            order.extend(sorted);
        }),
    }

    Ok(indices_array(order))
}

/// Stable ascending sort for a `DictArray`, returning original row indices.
///
/// The dictionary values are sorted once (O(dict log dict)); each code is remapped to
/// the sorted-rank of its value, and the output row permutation is a stable sort of the
/// remapped codes. Rows with null codes sort first (nulls-first order).
fn sort_dict_to_indices(dict: DictArray, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let n = dict.len();
    let values = dict.values().clone();
    let codes = dict.codes().clone();

    // Sort the dictionary values once; get the order of original value positions in sorted order.
    let value_order = sort_to_indices(&values, ctx)?;
    let value_order = value_order.execute::<PrimitiveArray>(ctx)?;

    let values_len = values.len();
    let mut rank = vec![0u32; values_len];
    for (sorted_pos, &orig_pos) in value_order.as_slice::<u32>().iter().enumerate() {
        // sorted_pos < values_len <= 2^32 (u32 row-index convention).
        #[allow(clippy::cast_possible_truncation, reason = "u32 row-index convention")]
        let rank_val = sorted_pos as u32;
        rank[orig_pos as usize] = rank_val;
    }

    // Remap codes: key[i] = rank[codes[i]] + 1 for valid rows, 0 for null rows.
    // Null codes sort first (key 0), then rows by ascending value rank.
    let codes_array = codes.execute::<PrimitiveArray>(ctx)?;
    let codes_mask = codes_array.validity()?.execute_mask(n, ctx)?;
    let mut keys: Vec<u32> = Vec::with_capacity(n);

    match_each_integer_ptype!(codes_array.ptype(), |I| {
        let code_values = codes_array.to_buffer::<I>();
        let mut iter = codes_mask.iter();
        for code in code_values.iter() {
            let is_valid = iter
                .next()
                .vortex_expect("code mask length must equal codes length");
            if is_valid {
                // Codes are u32-keyed by convention; the physical ptype may be
                // narrower/wider but rank is indexed by the u32 code value.
                #[allow(clippy::cast_possible_truncation, reason = "code values are u32-range by the dict-code convention")]
                let code_idx = *code as usize;
                keys.push(rank[code_idx] + 1);
            } else {
                keys.push(0);
            }
        }
    });

    // Stable sort of the keys gives the row permutation (radix on u32).
    Ok(indices_array(radix_sort_indices_u32(&keys)))
}



#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::ConstantArray;
    use crate::arrays::VarBinArray;
    use crate::assert_arrays_eq;

    fn indices_array_from_slice(indices: &[u32]) -> ArrayRef {
        PrimitiveArray::from_iter(indices.iter().copied()).into_array()
    }

    fn lcg(seed: u64) -> impl Iterator<Item = u64> {
        std::iter::successors(Some(seed), |&s| Some(s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)))
    }

    fn stable_reference<T: PartialOrd + Copy>(values: &[T]) -> Vec<u32> {
        // Test-only reference; u32 row-index convention.
        #[allow(clippy::cast_possible_truncation, reason = "u32 row-index convention")]
        let mut order: Vec<u32> = (0..values.len() as u32).collect();
        order.sort_by(|&a, &b| {
            let (va, vb) = (values[a as usize], values[b as usize]);
            // nulls handled by caller; here partial_cmp with total fallback
            if va == vb {
                a.cmp(&b)
            } else if va < vb {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        });
        order
    }

    #[test]
    fn constant_identity() -> VortexResult<()> {
        let arr = ConstantArray::new(42i32, 5).into_array();
        let mut ctx = array_session().create_execution_ctx();
        let idx = sort_to_indices(&arr, &mut ctx)?;
        let idx = idx.execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(idx.as_slice::<u32>(), &[0, 1, 2, 3, 4]);
        // Round-trip: take at identity returns the same constant data.
        let taken = arr.take(indices_array_from_slice(idx.as_slice()))?;
        assert_arrays_eq!(taken, arr, &mut ctx);
        Ok(())
    }

    #[test]
    fn primitive_sorted_identity() -> VortexResult<()> {
        let arr = buffer![1i32, 2, 3, 10, 100].into_array();
        let mut ctx = array_session().create_execution_ctx();
        let idx = sort_to_indices(&arr, &mut ctx)?;
        let idx = idx.execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(idx.as_slice::<u32>(), &[0, 1, 2, 3, 4]);
        Ok(())
    }

    #[test]
    fn primitive_unsorted() -> VortexResult<()> {
        let arr = buffer![3i32, 1, 2].into_array();
        let mut ctx = array_session().create_execution_ctx();
        let idx = sort_to_indices(&arr, &mut ctx)?;
        let idx = idx.execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(idx.as_slice::<u32>(), &[1, 2, 0]);
        Ok(())
    }

    #[test]
    fn primitive_dupes_stable() -> VortexResult<()> {
        // Duplicates keep original relative order.
        let arr = buffer![3u32, 1, 3, 2, 1].into_array();
        let mut ctx = array_session().create_execution_ctx();
        let idx = sort_to_indices(&arr, &mut ctx)?;
        let idx = idx.execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(idx.as_slice::<u32>(), &[1, 4, 3, 0, 2]);
        Ok(())
    }

    #[test]
    fn primitive_nulls_first() -> VortexResult<()> {
            let arr = PrimitiveArray::from_option_iter([Some(3i32), None, Some(1i32), None, Some(2i32)])
            .into_array();
        let mut ctx = array_session().create_execution_ctx();
        let idx = sort_to_indices(&arr, &mut ctx)?;
        let idx = idx.execute::<PrimitiveArray>(&mut ctx)?;
        // nulls first in original order: indices 1,3; then 1, 2, 3 ascending
        assert_eq!(idx.as_slice::<u32>(), &[1, 3, 2, 4, 0]);
        Ok(())
    }

    #[test]
    fn primitive_u64() -> VortexResult<()> {
        let arr = buffer![10u64, 9, 8, 7, 6].into_array();
        let mut ctx = array_session().create_execution_ctx();
        let idx = sort_to_indices(&arr, &mut ctx)?;
        let idx = idx.execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(idx.as_slice::<u32>(), &[4, 3, 2, 1, 0]);
        Ok(())
    }

    #[test]
    fn dict_values_sorted_codes_remapped() -> VortexResult<()> {
        // dict values: [3, 1, 2], codes: [0, 1, 2, 0, 1]
        // rows: 3, 1, 2, 3, 1
        // values sorted: [1, 2, 3] -> value_order [1, 2, 0]
        // rank = [2, 0, 1]; keys = [3, 1, 2, 3, 1]
        // row order = stable sort of keys = [1, 4, 2, 0, 3]
        // materialized rows at that order: 1, 1, 2, 3, 3 (sorted)
        let values = buffer![3i32, 1, 2].into_array();
        let codes = buffer![0u32, 1, 2, 0, 1].into_array();
        let dict = DictArray::try_new(codes, values)?.into_array();

        let mut ctx = array_session().create_execution_ctx();
        let idx = sort_to_indices(&dict, &mut ctx)?;
        let idx = idx.execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(idx.as_slice::<u32>(), &[1, 4, 2, 0, 3]);

        // Materialization equals a sorted reference.
        let taken = dict.take(indices_array_from_slice(idx.as_slice()))?;
        let reference = buffer![1i32, 1, 2, 3, 3].into_array();
        assert_arrays_eq!(taken, reference, &mut ctx);
        Ok(())
    }

    #[test]
    fn dict_of_strings_unsupported() -> VortexResult<()> {
        let values = VarBinArray::from(vec!["c", "a", "b"]).into_array();
        let codes = buffer![0u32, 1, 2, 0, 1].into_array();
        let dict = DictArray::try_new(codes, values)?.into_array();
        let mut ctx = array_session().create_execution_ctx();
        let err = sort_to_indices(&dict, &mut ctx)
            .err()
            .vortex_expect("expected NotImplemented for string dict values");
        assert!(err.to_string().contains("not implemented"), "{err}");
        Ok(())
    }

    #[test]
    fn dict_materialization_round_trip() -> VortexResult<()> {
        let values = buffer![30i32, 10, 20].into_array();
        let codes = buffer![0u32, 1, 2, 1, 0].into_array();
        let dict = DictArray::try_new(codes, values)?.into_array();

        let mut ctx = array_session().create_execution_ctx();
        let idx = sort_to_indices(&dict, &mut ctx)?;
        let idx = idx.execute::<PrimitiveArray>(&mut ctx)?;

        // take at the returned indices must equal the sorted canonical reference.
        let taken = dict.take(indices_array_from_slice(idx.as_slice()))?;
        let reference = buffer![10i32, 10, 20, 30, 30].into_array();
        assert_arrays_eq!(taken, reference, &mut ctx);
        Ok(())
    }

    #[test]
    fn dict_null_codes_sort_first() -> VortexResult<()> {
            let values = buffer![30i32, 10, 20].into_array();
        let codes = PrimitiveArray::from_option_iter([Some(0u32), None, Some(1u32), Some(2u32), None])
            .into_array();
        let dict = DictArray::try_new(codes, values)?.into_array();

        let mut ctx = array_session().create_execution_ctx();
        let idx = sort_to_indices(&dict, &mut ctx)?;
        let idx = idx.execute::<PrimitiveArray>(&mut ctx)?;

        let taken = dict.take(indices_array_from_slice(idx.as_slice()))?;
        let reference =
            PrimitiveArray::from_option_iter([None, None, Some(10i32), Some(20), Some(30)])
                .into_array();
        assert_arrays_eq!(taken, reference, &mut ctx);
        Ok(())
    }

    #[test]
    fn round_trip_take_is_sorted() -> VortexResult<()> {
        let arr = buffer![5i32, 3, 9, 1, 7, 3].into_array();
        let mut ctx = array_session().create_execution_ctx();
        let idx = sort_to_indices(&arr, &mut ctx)?;
        let idx = idx.execute::<PrimitiveArray>(&mut ctx)?;
        let taken = arr.take(indices_array_from_slice(idx.as_slice()))?;
        let expected = buffer![1i32, 3, 3, 5, 7, 9].into_array();
        assert_arrays_eq!(taken, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn randomized_lcg() -> VortexResult<()> {
        let mut rng = lcg(0xdead_beef);
        let values: Vec<u32> = rng.by_ref().take(256).map(|x| (x % 1000) as u32).collect();
        let arr = PrimitiveArray::from_iter(values.clone()).into_array();

        let mut ctx = array_session().create_execution_ctx();
        let idx = sort_to_indices(&arr, &mut ctx)?;
        let idx = idx.execute::<PrimitiveArray>(&mut ctx)?;

        let reference = stable_reference(&values);
        assert_eq!(idx.as_slice::<u32>(), reference.as_slice());

        // Round-trip check as well.
        let taken = arr.take(indices_array_from_slice(idx.as_slice()))?;
        let mut sorted: Vec<u32> = values.clone();
        sorted.sort();
        for i in 0..values.len() {
            let expect_val = sorted[i];
            let scalar = taken.execute_scalar(i, &mut ctx)?.as_primitive().typed_value::<u32>();
            assert_eq!(scalar, Some(expect_val));
        }
        Ok(())
    }

    #[test]
    fn unsupported_non_numeric_is_error() -> VortexResult<()> {
        let arr = VarBinArray::from(vec!["a", "b", "c"]).into_array();
        let mut ctx = array_session().create_execution_ctx();
        let err = sort_to_indices(&arr, &mut ctx).err().vortex_expect("expected error");
        assert!(err.to_string().contains("not implemented"), "{err}");
        Ok(())
    }

    #[test]
    fn is_sorted_stat_short_circuit() -> VortexResult<()> {
        use crate::expr::stats::Precision;
        use crate::expr::stats::Stat;

        let arr = buffer![1i32, 2, 3].into_array();
        // Pre-seed the IsSorted stat to prove the short-circuit path returns identity.
        arr.statistics().set(Stat::IsSorted, Precision::Exact(true.into()));
        let mut ctx = array_session().create_execution_ctx();
        let idx = sort_to_indices(&arr, &mut ctx)?;
        let idx = idx.execute::<PrimitiveArray>(&mut ctx)?;
        assert_eq!(idx.as_slice::<u32>(), &[0, 1, 2]);
        Ok(())
    }
}

/// The sortedness of an array (flatland §3.3).
///
/// `StrictSorted` implies `Sorted`. Computed through the cached
/// `Stat::IsSorted`/`Stat::IsStrictSorted` statistics — an exact cached value
/// is O(1); otherwise the accumulator computes it once and the stat is cached
/// for the next reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sortedness {
    /// Ascending, no duplicates.
    StrictSorted,
    /// Ascending (duplicates allowed).
    Sorted,
    /// Known unsorted, or unsupported dtype (struct/list/FSL report Unsorted).
    Unsorted,
}

/// Return the array's sortedness, preferring cached exact stats (O(1)) and
/// computing+caching otherwise. This is the flatland SortKernel short-circuit:
/// `Sorted | StrictSorted` ⇒ the identity permutation.
pub fn sortedness(arr: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Sortedness> {
    use crate::aggregate_fn::fns::is_sorted::{is_sorted, is_strict_sorted};
    if is_strict_sorted(arr, ctx)? {
        return Ok(Sortedness::StrictSorted);
    }
    if is_sorted(arr, ctx)? {
        return Ok(Sortedness::Sorted);
    }
    Ok(Sortedness::Unsorted)
}

#[cfg(test)]
mod sortedness_tests {
    use vortex_buffer::buffer;

    use super::*;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;

    #[test]
    fn sortedness_tri_state() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let strict = buffer![1u32, 2, 3].into_array();
        let dupes = buffer![1u32, 2, 2].into_array();
        let unsorted = buffer![3u32, 1, 2].into_array();
        assert_eq!(sortedness(&strict, &mut ctx)?, Sortedness::StrictSorted);
        assert_eq!(sortedness(&dupes, &mut ctx)?, Sortedness::Sorted);
        assert_eq!(sortedness(&unsorted, &mut ctx)?, Sortedness::Unsorted);
        // Second call hits the O(1) cached stat path.
        assert_eq!(sortedness(&unsorted, &mut ctx)?, Sortedness::Unsorted);
        Ok(())
    }
}
