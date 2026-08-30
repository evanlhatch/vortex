// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Tests for the encoded-domain affine transform (REBUILD Part 0: the 3
// sound arms — Constant rewrite, FoR ref-bump, Dict values-map — plus the
// Primitive row loop). Every arm must decode equal to the canonical affine.


// Integration-test crate: every fn is a test; the legacy session helper is
// the established flatland test fixture.
#![allow(clippy::tests_outside_test_module)]
#![allow(clippy::disallowed_methods, reason = "legacy_session is the flatland test fixture")]
#![allow(clippy::min_ident_chars, reason = "short names are idiomatic in test bodies")]


#![allow(clippy::cast_possible_truncation, reason = "flatland u32-key convention in tests/benches")]
#![allow(clippy::redundant_clone, reason = "test fixtures; clarity over micro-optimization")]

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt as _;
use vortex_array::validity::Validity;
use vortex_array::builders::dict::dict_encode;
use vortex_error::VortexExpect as _;

fn u32_arr(values: &[u32]) -> vortex_array::ArrayRef {
    PrimitiveArray::new(values.to_vec(), Validity::NonNullable).into_array()
}

fn decoded(arr: &vortex_array::ArrayRef, ctx: &mut vortex_array::ExecutionCtx) -> Vec<u32> {
    arr.clone()
        .execute::<PrimitiveArray>(ctx)
        .vortex_expect("execute")
        .as_slice::<u32>()
        .to_vec()
}

#[test]
fn affine_for_add_is_ref_bump() {
    let ctx = &mut vortex_array::legacy_session().create_execution_ctx();
    let src = u32_arr(&[1000, 1001, 1002, 1003, 1004, 1005]);
    let parray = src
        .clone()
        .downcast::<vortex_array::arrays::Primitive>();
    let for_arr = vortex_fastlanes::FoR::encode(parray, ctx).vortex_expect("for encode");
    let arr = for_arr.into_array();
    assert!(arr.is::<vortex_fastlanes::FoR>(), "fixture must be FoR");

    let out = vortex_fastlanes::affine::affine(&arr, 1, 5, ctx).vortex_expect("affine");

    // Tier-1 win: the result is STILL a FoR (reference bumped, no row work).
    assert!(out.is::<vortex_fastlanes::FoR>(), "add must keep FoR encoding");
    let parts: vortex_fastlanes::raw_parts::FoRParts =
        <vortex_fastlanes::FoR as vortex_array::raw_parts::RawParts<u32>>::raw_parts(&out)
            .vortex_expect("for parts after affine");
    assert_eq!(
        parts.reference,
        vortex_array::scalar::PValue::U32(1005),
        "reference must be bumped by base"
    );

    // Decodes equal to canonical +5.
    assert_eq!(decoded(&out, ctx), vec![1005, 1006, 1007, 1008, 1009, 1010]);
}

#[test]
fn affine_dict_mul_maps_values() {
    let ctx = &mut vortex_array::legacy_session().create_execution_ctx();
    let keys = u32_arr(&[7, 9, 7, 9, 7]);
    let dict = dict_encode(&keys, ctx).vortex_expect("dict");
    let dict_arr = dict.into_array();
    assert!(
        dict_arr.is::<vortex_array::arrays::Dict>(),
        "fixture must be Dict"
    );

    let out = vortex_fastlanes::affine::affine(&dict_arr, 2, 0, ctx).vortex_expect("affine");
    // O(|dict|): codes reused, values mapped — encoding stays Dict.
    assert!(out.is::<vortex_array::arrays::Dict>(), "mul must keep Dict encoding");
    assert_eq!(decoded(&out, ctx), vec![14, 18, 14, 18, 14]);
}

#[test]
fn affine_dict_add_maps_values() {
    let ctx = &mut vortex_array::legacy_session().create_execution_ctx();
    let keys = u32_arr(&[10, 20, 10, 30]);
    let dict = dict_encode(&keys, ctx).vortex_expect("dict");
    let dict_arr = dict.into_array();

    let out = vortex_fastlanes::affine::affine(&dict_arr, 1, 5, ctx).vortex_expect("affine");
    assert_eq!(decoded(&out, ctx), vec![15, 25, 15, 35]);
}

#[test]
fn affine_constant_is_scalar_rewrite() {
    let ctx = &mut vortex_array::legacy_session().create_execution_ctx();
    let arr = ConstantArray::new(7u32, 5usize).into_array();
    let out = vortex_fastlanes::affine::affine(&arr, 3, 1, ctx).vortex_expect("affine");
    assert!(
        out.is::<vortex_array::arrays::Constant>(),
        "affine must keep Constant encoding"
    );
    assert_eq!(decoded(&out, ctx), vec![22; 5]);
}

#[test]
fn affine_primitive_row_loop() {
    let ctx = &mut vortex_array::legacy_session().create_execution_ctx();
    let src = u32_arr(&[1, 2, 3, 4, 5, 100, 0, u32::MAX - 1]);
    // factor=2 exercises the wrapping row loop; add-only rides the portable
    // SIMD tier.
    let out = vortex_fastlanes::affine::affine(&src, 2, 0, ctx).vortex_expect("affine mul");
    assert_eq!(
        decoded(&out, ctx),
        // (u32::MAX-1)*2 wraps to 2^32-4 — modular arithmetic, by design.
        vec![2, 4, 6, 8, 10, 200, 0, 4294967292]
    );

    let out = vortex_fastlanes::affine::affine(&src, 1, 5, ctx).vortex_expect("affine add");
    assert_eq!(
        decoded(&out, ctx),
        vec![6, 7, 8, 9, 10, 105, 5, 3]
    );
}

#[test]
fn affine_fo_scale_falls_to_primitive() {
    let ctx = &mut vortex_array::legacy_session().create_execution_ctx();
    let src = u32_arr(&[1000, 1001, 1002]);
    let parray = src.clone().downcast::<vortex_array::arrays::Primitive>();
    let for_arr = vortex_fastlanes::FoR::encode(parray, ctx).vortex_expect("for encode");
    let arr = for_arr.into_array();

    let out = vortex_fastlanes::affine::affine(&arr, 2, 0, ctx).vortex_expect("affine");
    // Scale widens bit-width → decompress to Primitive (re-pack is T2.2).
    assert!(
        out.is::<vortex_array::arrays::Primitive>(),
        "scale must fall to the primitive path"
    );
    assert_eq!(decoded(&out, ctx), vec![2000, 2002, 2004]);
}

#[test]
fn affine_scalar_overflow_is_error() {
    let ctx = &mut vortex_array::legacy_session().create_execution_ctx();
    // u8 constant 200 * 2 overflows u8 → error, not a silent wrap.
    let arr = ConstantArray::new(200u8, 3usize).into_array();
    assert!(
        vortex_fastlanes::affine::affine(&arr, 2, 0, ctx).is_err(),
        "ptype overflow must error"
    );
}

#[test]
fn affine_rejects_float() {
    let ctx = &mut vortex_array::legacy_session().create_execution_ctx();
    let arr = PrimitiveArray::new(vec![1.0f64, 2.0], Validity::NonNullable).into_array();
    assert!(
        vortex_fastlanes::affine::affine(&arr, 2, 0, ctx).is_err(),
        "floats never ride affine (flatland: f* via ALP)"
    );
}
