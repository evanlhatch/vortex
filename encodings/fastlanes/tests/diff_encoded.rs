// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Tests for diff_encoded (REBUILD Part 3 #5 extension): same-reference FoR
// pairs diff their encoded children; the result must equal the generic
// diff semantics (patches over `new` at every differing row).

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt as _;
use vortex_array::flatland::verbs;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect as _;
use vortex_fastlanes::flatland::diff::diff_encoded;

fn u32_arr(values: &[u32]) -> vortex_array::ArrayRef {
    PrimitiveArray::new(values.to_vec(), Validity::NonNullable).into_array()
}

fn for_encode(values: &[u32], ctx: &mut vortex_array::ExecutionCtx) -> vortex_array::ArrayRef {
    let arr = u32_arr(values);
    let p = arr.downcast::<vortex_array::arrays::Primitive>();
    vortex_fastlanes::FoR::encode(p, ctx)
        .vortex_expect("for encode")
        .into_array()
}

fn patch_contents(p: &Patches, ctx: &mut vortex_array::ExecutionCtx) -> (Vec<u32>, Vec<u32>) {
    let indices = p
        .indices()
        .clone()
        .execute::<PrimitiveArray>(ctx)
        .vortex_expect("indices execute");
    let values = p
        .values()
        .clone()
        .execute::<PrimitiveArray>(ctx)
        .vortex_expect("values execute");
    (
        indices.as_slice::<u32>().to_vec(),
        values.as_slice::<u32>().to_vec(),
    )
}

#[test]
fn diff_encoded_same_ref_for_matches_generic() {
    let ctx = &mut vortex_array::legacy_session().create_execution_ctx();

    // Both sides FoR-encoded with the SAME reference (min 1000) — children
    // are directly comparable, zero canonicalization.
    let old = for_encode(&[1000, 1001, 1002, 1003, 1004], ctx);
    let new = for_encode(&[1000, 1009, 1002, 1003, 1010], ctx);
    assert!(old.is::<vortex_fastlanes::FoR>() && new.is::<vortex_fastlanes::FoR>());

    let got = diff_encoded(&old, &new, ctx).vortex_expect("diff_encoded");
    let want = verbs::diff(&old, &new, ctx).vortex_expect("generic diff");

    assert_eq!(got.array_len(), 5);
    assert_eq!(
        patch_contents(&got, ctx),
        patch_contents(&want, ctx),
        "encoded diff must equal generic diff"
    );
    assert_eq!(patch_contents(&got, ctx).0, vec![1, 4]);
    assert_eq!(patch_contents(&got, ctx).1, vec![1009, 1010]);
}

#[test]
fn diff_encoded_identical_columns_empty_patches() {
    let ctx = &mut vortex_array::legacy_session().create_execution_ctx();
    let old = for_encode(&[500, 501, 502], ctx);
    let new = for_encode(&[500, 501, 502], ctx);
    let got = diff_encoded(&old, &new, ctx).vortex_expect("diff_encoded");
    assert_eq!(got.num_patches(), 0, "no changes → empty patches");
}

#[test]
fn diff_encoded_differing_refs_falls_through() {
    let ctx = &mut vortex_array::legacy_session().create_execution_ctx();
    // Different references: children are NOT comparable — must fall through
    // to the generic path and still produce the right answer.
    let old = for_encode(&[1000, 1001, 1002], ctx);
    let new = for_encode(&[2000, 2001, 2003], ctx);

    let got = diff_encoded(&old, &new, ctx).vortex_expect("diff_encoded");
    let want = verbs::diff(&old, &new, ctx).vortex_expect("generic diff");
    assert_eq!(
        patch_contents(&got, ctx),
        patch_contents(&want, ctx),
        "differing refs must ride the generic path with identical results"
    );
    assert_eq!(patch_contents(&got, ctx).0, vec![0, 1, 2]);
}

#[test]
fn diff_encoded_mixed_encodings_fall_through() {
    let ctx = &mut vortex_array::legacy_session().create_execution_ctx();
    // FoR vs plain Primitive: generic path handles it.
    let old = for_encode(&[10, 11, 12], ctx);
    let new = u32_arr(&[10, 99, 12]);
    let got = diff_encoded(&old, &new, ctx).vortex_expect("diff_encoded");
    assert_eq!(patch_contents(&got, ctx).0, vec![1]);
}
