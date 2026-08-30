// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Encoded-diff wiring (flatland REBUILD Part 3 #5 extension): change
//! detection runs ON encoded columns — no canonicalize.
//!
//! The generic `vortex_array::flatland::verbs::diff` already rides the
//! encoding-native compare kernels (FoR/ALP/BitPacked all ship NotEq
//! compares upstream) → mask → indices → encoding-native `take` → Patches.
//! This module adds the tier-1 fork win on top:
//!
//! **Same-reference FoR pairs** compare their encoded children directly
//! through the u32 portable/SVE neq path. Soundness: FoR decode is
//! `v = child + ref` (wrapping), and wrapping add is injective for a fixed
//! ref, so `a == b ⟺ child_a == child_b` — but ONLY when both sides share
//! the reference. Differing refs fall through to the generic path.
//!
//! Patch values are promoted back to the logical domain (`delta + ref`,
//! wrapping) via the portable SIMD add-const tier — patches carry `new`'s
//! LOGICAL values, never child deltas.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::flatland::verbs::diff as diff_generic;
use vortex_array::flatland::verbs::diff_fast_path;
use vortex_array::patches::Patches;
use vortex_array::scalar::PValue;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;

use crate::FoR;
use crate::FoRArrayExt as _;
use crate::FoRSlots;

/// Change detection on encoded columns (same contract as
/// [`vortex_array::flatland::verbs::diff`]: patches over `new` at every row
/// where the two differ; applying them to `old` yields `new`).
pub fn diff_encoded(
    old: &ArrayRef,
    new: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Patches> {
    if let Some(patches) = diff_for_same_ref(old, new, ctx) {
        return Ok(patches);
    }
    // Generic path: encoding-native NotEq compare → mask → indices →
    // encoding-native take from `new`. No canonicalize on the compare or
    // take (FoR/BitPacked compute kernels are encoding-native upstream).
    diff_generic(old, new, ctx)
}

/// Same-reference FoR fast path: diff the encoded children with the u32
/// SVE/portable neq path. Returns `None` unless both sides are FoR with
/// byte-equal references and u32 primitive encoded children.
fn diff_for_same_ref(
    old: &ArrayRef,
    new: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> Option<Patches> {
    if old.len() != new.len() || old.dtype() != new.dtype() {
        return None;
    }
    if !old.is::<FoR>() || !new.is::<FoR>() {
        return None;
    }
    let old_for = old.clone().downcast::<FoR>();
    let new_for = new.clone().downcast::<FoR>();
    // References must match exactly (dtype + value) for the children to be
    // comparable; PValue carries both, None on either side falls through.
    let old_ref = old_for.reference_scalar().as_primitive().pvalue();
    let new_ref = new_for.reference_scalar().as_primitive().pvalue();
    if old_ref.is_none() || old_ref != new_ref {
        return None;
    }
    let reference = match old_ref {
        // Children are u32 (checked below), so the reference must be u32 —
        // any other width falls to the generic path.
        Some(PValue::U32(r)) => r,
        _ => return None,
    };
    let oc = fo_r_encoded_child(old)?;
    let nc = fo_r_encoded_child(new)?;
    // Children must be host-resident u32 primitives — the SVE fast path's
    // exact operand contract; anything else falls to the generic path.
    if oc.as_opt::<Primitive>().is_none() || nc.as_opt::<Primitive>().is_none() {
        return None;
    }
    let child_patches = diff_fast_path(&oc, &nc)?;
    // Unchanged columns short-circuit before any delta promotion.
    if child_patches.num_patches() == 0 {
        return Some(child_patches);
    }
    // Promote child deltas to LOGICAL values: v = delta + ref (wrapping).
    // Patches over a FoR column must carry the column's logical dtype.
    let deltas = child_patches
        .values()
        .clone()
        .execute::<PrimitiveArray>(ctx)
        .ok()?;
    let mut logical = vec![0u32; deltas.len()];
    vortex_buffer::portable::add_const_u32(
        deltas.as_slice::<u32>(),
        reference,
        &mut logical,
    );
    let promoted = vortex_array::IntoArray::into_array(PrimitiveArray::new(
        logical,
        Validity::NonNullable,
    ));
    // SAFETY: identical shape to `child_patches` (same indices, same array
    // len); only the values were widened back to the logical domain.
    let promoted = unsafe {
        Patches::new_simple_unchecked(
            child_patches.array_len(),
            0,
            child_patches.indices().clone(),
            promoted,
        )
    };
    Some(promoted)
}

fn fo_r_encoded_child(array: &ArrayRef) -> Option<ArrayRef> {
    array
        .slots()
        .get(FoRSlots::ENCODED)
        .and_then(|s| s.as_ref())
        .cloned()
}
