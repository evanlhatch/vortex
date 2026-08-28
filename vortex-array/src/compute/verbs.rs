// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Flatland fork verbs (REBUILD Part 3 #5, #6, #7): change detection, indexed
//! write, and grouping as first-class array operations.
//!
//! Placement note: the REBUILD sketch put these in vortex-compute, but 0.85's
//! vortex-compute is a buffer/lane-level crate with no ArrayRef dependency —
//! array-level verbs live here instead.

use vortex_error::vortex_err;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::Primitive;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::builders::dict::dict_encode;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::patches::Patches;
use crate::scalar_fn::fns::binary::execute_compare;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::validity::Validity;

fn u32_array(values: &[u32]) -> ArrayRef {
    PrimitiveArray::new(values.to_vec(), Validity::NonNullable).into_array()
}

/// Change detection as a primitive (REBUILD Part 3 #5):
/// `diff(old, new)` → patches over `new` at every row where the two differ.
///
/// The returned [`Patches`] carries `new`'s values at changed rows; applying
/// them to `old` yields `new`. An identical pair yields empty patches.
pub fn diff(old: &ArrayRef, new: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Patches> {
    vortex_ensure!(
        old.len() == new.len(),
        "diff requires equal lengths, got {} and {}",
        old.len(),
        new.len()
    );
    vortex_ensure!(
        old.dtype() == new.dtype(),
        "diff requires equal dtypes, got {:?} and {:?}",
        old.dtype(),
        new.dtype()
    );

    let changed = execute_compare(old, new, CompareOperator::NotEq, ctx)?;
    let mask = Mask::from_buffer(changed.execute::<BoolArray>(ctx)?.into_bit_buffer());

    // Identical arrays: zero-length patch arrays (Patches::new rejects empties).
    if matches!(mask.indices(), AllOr::None) {
        return unsafe {
            Ok(Patches::new_simple_unchecked(
                new.len(),
                0,
                u32_array(&[]),
                new.clone(),
            ))
        };
    }

    let indices: Vec<u32> = match mask.indices() {
        AllOr::All => (0..new.len() as u32).collect(),
        AllOr::Some(idx) => idx.iter().map(|&i| i as u32).collect(),
        AllOr::None => unreachable!("handled above"),
    };

    let values = new.take(u32_array(&indices))?;
    unsafe { Ok(Patches::new_simple_unchecked(new.len(), 0, u32_array(&indices), values)) }
}

/// Indexed write (REBUILD Part 3 #6): `result[indices[i]] = values[i]` for
/// every i. Returns a new array; the input is never mutated.
///
/// Only primitive dtypes are supported — this verb exists for the flatland
/// overlay write path, whose resting state is primitive-typed columns.
pub fn scatter(
    target: &ArrayRef,
    indices: &ArrayRef,
    values: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let mut owned = target.clone().execute::<PrimitiveArray>(ctx)?.into_array();
    scatter_in_place(&mut owned, indices, values, ctx)?;
    Ok(owned)
}

/// In-place variant of [`scatter`] on a uniquely-owned primitive array —
/// no clone, no realloc when the backing buffer is uniquely held (REBUILD
/// ownership model: take-once-per-column-per-tick).
pub fn scatter_in_place(
    target: &mut ArrayRef,
    indices: &ArrayRef,
    values: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let len = target.len();
    let ptype = target
        .as_opt::<Primitive>()
        .map(|p| p.ptype())
        .ok_or_else(|| vortex_err!("scatter requires a primitive-typed target"))?;

    let indices_arr = indices.clone().execute::<PrimitiveArray>(ctx)?;
    let values_arr = values.clone().execute::<PrimitiveArray>(ctx)?;
    vortex_ensure!(
        indices_arr.ptype() == PType::U32,
        "scatter indices must be u32, got {:?}",
        indices_arr.ptype()
    );
    vortex_ensure!(
        values_arr.ptype() == ptype,
        "scatter value ptype {:?} != target ptype {:?}",
        values_arr.ptype(),
        ptype
    );
    vortex_ensure!(
        indices_arr.len() == values_arr.len(),
        "scatter indices/values length mismatch: {} vs {}",
        indices_arr.len(),
        values_arr.len()
    );

    let idx_slice = indices_arr.as_slice::<u32>();
    let mut ok_bounds = true;
    for &i in idx_slice {
        ok_bounds &= (i as usize) < len;
    }
    vortex_ensure!(ok_bounds, "scatter index out of bounds for len {}", len);

    // Fast path: uniquely-owned buffer — in-place atomic writes.
    let mut written = false;
    match_each_native_ptype!(ptype, |T| {
        let vals = values_arr.as_slice::<T>();
        if let Some(mut guard) = target.try_buffer_mut::<T>() {
            for (slot, &i) in idx_slice.iter().enumerate() {
                guard[i as usize] = vals[slot];
            }
            written = true;
        }
    });
    if written {
        return Ok(());
    }

    // Rebase-with-intent (REBUILD Part 5): buffer shared — copy and overwrite.
    rebase_scatter(target, idx_slice, &values_arr, ctx)
}


/// Rebase path: materialize a fresh copy of `target` with the scatter applied,
/// then replace the target array. Only reached when in-place mutation is
/// impossible (shared buffer).
fn rebase_scatter(
    target: &mut ArrayRef,
    idx: &[u32],
    values_arr: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    use crate::arrays::primitive::PrimitiveArrayExt as _;
    let ptype = target
        .as_opt::<Primitive>()
        .map(|p| p.ptype())
        .ok_or_else(|| vortex_err!("scatter requires a primitive-typed target"))?;
    let base = target.clone().execute::<PrimitiveArray>(ctx)?;
    let validity = base.validity()?;

    match_each_native_ptype!(ptype, |T| {
        let mut out = base.as_slice::<T>().to_vec();
        let vals = values_arr.as_slice::<T>().to_vec();
        for (slot, &i) in idx.iter().enumerate() {
            out[i as usize] = vals[slot];
        }
        *target = PrimitiveArray::new(out, validity.clone()).into_array();
    });
    Ok(())
}

/// Grouping (REBUILD Part 3 #7): assign each row a dense group code via
/// dictionary encoding (codes come free), then counting-sort the codes into
/// per-group lengths.
///
/// Group ids follow first-appearance order of the distinct key values.
pub fn group_indices(keys: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<GroupIndices> {
    let dict = dict_encode(keys, ctx)?;
    let num_groups = dict.values().len();

    let codes_arr = dict.codes().clone();
    // Dict codes may be any unsigned ptype (utf8 dicts encode u8 codes);
    // normalize to u32 so group codes are dense and u32-typed.
    let codes_arr = match codes_arr.dtype() {
        DType::Primitive(PType::U32 | PType::U64, _) => codes_arr,
        _ => codes_arr.cast(DType::Primitive(PType::U32, Nullability::NonNullable))?,
    };
    let codes = codes_arr.execute::<PrimitiveArray>(ctx)?;
    let code_slice = codes.as_slice::<u32>();

    let mut group_lengths = vec![0u32; num_groups];
    for &c in code_slice {
        group_lengths[c as usize] += 1;
    }

    Ok(GroupIndices {
        codes: codes.into_array(),
        distinct_values: dict.values().clone(),
        group_lengths,
    })
}

/// Result of [`group_indices`]: per-row group codes, the distinct key values
/// backing those codes, and per-group row counts.
#[derive(Debug, Clone)]
pub struct GroupIndices {
    /// `codes[i]` = group id of row i (first-appearance order).
    pub codes: ArrayRef,
    /// Distinct key values; `distinct_values[code]` is the group's key.
    pub distinct_values: ArrayRef,
    /// Number of rows in each group.
    pub group_lengths: Vec<u32>,
}
