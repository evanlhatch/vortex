// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Tests for the fastlanes RawParts impls (REBUILD Part 3 #3).
// Builds FoR + BitPacked arrays and decodes through the parts.


// Integration-test crate: every fn is a test; the legacy session helper is
// the established flatland test fixture.
#![allow(clippy::tests_outside_test_module)]
#![allow(clippy::disallowed_methods, reason = "legacy_session is the flatland test fixture")]
#![allow(clippy::min_ident_chars, reason = "short names are idiomatic in test bodies")]


#![allow(clippy::cast_possible_truncation, reason = "flatland u32-key convention in tests/benches")]
#![allow(clippy::redundant_clone, reason = "test fixtures; clarity over micro-optimization")]

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::raw_parts::RawParts;
use vortex_error::VortexExpect as _;
use vortex_fastlanes::raw_parts::{BitPackedParts, FoRParts};

fn i64_arr(values: &[i64]) -> vortex_array::ArrayRef {
    PrimitiveArray::new(values.to_vec(), vortex_array::validity::Validity::NonNullable)
        .into_array()
}

#[test]
fn for_raw_parts_decode_roundtrip() {
    let ctx = &mut vortex_array::legacy_session().create_execution_ctx();
    let src = i64_arr(&[1000, 1001, 1002, 1003, 1004, 1005]);
    let parray = match src.clone().try_downcast::<vortex_array::arrays::Primitive>() {
        Ok(a) => a,
        Err(e) => panic!("downcast failed: {e:?}"),
    };
    let encoded = vortex_fastlanes::FoR::encode(parray, ctx)
    .vortex_expect("for encode");
    let arr = encoded.into_array();

    let parts: FoRParts = <vortex_fastlanes::FoR as RawParts<i64>>::raw_parts(&arr)
        .vortex_expect("for parts");
    // Reference is the frame (1000); encoded child holds deltas.
    assert_eq!(
        parts.reference,
        vortex_array::scalar::PValue::I64(1000)
    );
    let encoded_child = parts.encoded.clone().execute::<PrimitiveArray>(ctx).vortex_expect("exec encoded");
    // Deltas should be 0..5 (or whatever compression chose); decode via ref+delta
    // reconstruct the source values exactly.
    let decoded: Vec<i64> = encoded_child
        .as_slice::<i64>()
        .iter()
        .map(|d| *d + 1000)
        .collect();
    assert_eq!(decoded, vec![1000, 1001, 1002, 1003, 1004, 1005]);

    // Wrong ptype → None.
    assert!(
        <vortex_fastlanes::FoR as RawParts<u32>>::raw_parts(&arr).is_none(),
        "i64 FoR must not expose u32 parts"
    );
}

#[test]
fn bitpacked_raw_parts_expose_packed_words() {
    let ctx = &mut vortex_array::legacy_session().create_execution_ctx();
    // Values that fit in 4 bits.
    let src = i64_arr(&[0, 1, 15, 8, 3, 7, 12, 9, 2, 4]);
    let encoded = vortex_fastlanes::BitPacked::encode(&src, 4, ctx).vortex_expect("bp encode");
    let arr = encoded.into_array();

    let parts: BitPackedParts = <vortex_fastlanes::BitPacked as RawParts<i64>>::raw_parts(&arr)
        .vortex_expect("bp parts");
    assert_eq!(parts.bit_width, 4);
    assert!(parts.packed_len > 0);
    assert!(!parts.packed.is_null());
}
