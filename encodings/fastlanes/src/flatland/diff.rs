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

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Primitive;
use vortex_array::flatland::verbs::diff as diff_generic;
use vortex_array::flatland::verbs::diff_fast_path;
use vortex_array::patches::Patches;
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
    if let Some(patches) = diff_for_same_ref(old, new) {
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
fn diff_for_same_ref(old: &ArrayRef, new: &ArrayRef) -> Option<Patches> {
    if old.len() != new.len() || old.dtype() != new.dtype() {
        return None;
    }
    if !old.is::<FoR>() || !new.is::<FoR>() {
        return None;
    }
    let o = old.clone().downcast::<FoR>();
    let n = new.clone().downcast::<FoR>();
    // References must match exactly (dtype + value) for the children to be
    // comparable; PValue carries both, None on either side falls through.
    let o_ref = o.reference_scalar().as_primitive().pvalue();
    let n_ref = n.reference_scalar().as_primitive().pvalue();
    if o_ref.is_none() || o_ref != n_ref {
        return None;
    }
    let oc = fo_r_encoded_child(old)?;
    let nc = fo_r_encoded_child(new)?;
    // Children must be host-resident u32 primitives — the SVE fast path's
    // exact operand contract; anything else falls to the generic path.
    if oc.as_opt::<Primitive>().is_none() || nc.as_opt::<Primitive>().is_none() {
        return None;
    }
    diff_fast_path(&oc, &nc)
}

fn fo_r_encoded_child(array: &ArrayRef) -> Option<ArrayRef> {
    array
        .slots()
        .get(FoRSlots::ENCODED)
        .and_then(|s| s.as_ref())
        .cloned()
}
