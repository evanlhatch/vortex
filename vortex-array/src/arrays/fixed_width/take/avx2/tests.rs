// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![cfg_attr(miri, ignore)]
#![cfg(target_arch = "x86_64")]

use super::take_avx2;

macro_rules! test_cases {
    (index_type => $IDX:ty, value_types => $($VAL:ty),+) => {
        paste::paste! {
            $(
                #[test]
                #[allow(clippy::cast_possible_truncation)]
                fn [<test_avx2_take_simple_ $IDX _ $VAL>]() {
                    let values: Vec<$VAL> = (1..=127).map(|x| x as $VAL).collect();
                    let indices: Vec<$IDX> = (0..127).collect();

                    let result = unsafe { take_avx2(&values, &indices) };
                    assert_eq!(&values, result.as_slice());
                }

                #[test]
                #[should_panic(expected = "cannot take a non-empty set of indices")]
                #[allow(clippy::cast_possible_truncation)]
                fn [<test_avx2_take_empty_ $IDX _ $VAL>]() {
                    let values: Vec<$VAL> = vec![];
                    let indices: Vec<$IDX> = (0..127).collect();
                    drop(unsafe { take_avx2(&values, &indices) });
                }

                #[test]
                #[should_panic(expected = "take index out of bounds")]
                #[allow(clippy::cast_possible_truncation)]
                fn [<test_avx2_take_invalid_ $IDX _ $VAL>]() {
                    let values: Vec<$VAL> = (1..=127).map(|x| x as $VAL).collect();
                    let indices: Vec<$IDX> = (127..=254).collect();
                    drop(unsafe { take_avx2(&values, &indices) });
                }
            )+
        }
    };
}

test_cases!(
    index_type => u8,
    value_types => u32, i32, u64, i64, f32, f64
);
test_cases!(
    index_type => u16,
    value_types => u32, i32, u64, i64, f32, f64
);
test_cases!(
    index_type => u32,
    value_types => u32, i32, u64, i64, f32, f64
);
test_cases!(
    index_type => u64,
    value_types => u32, i32, u64, i64, f32, f64
);

#[test]
fn last_valid_u8_index() {
    let values: Vec<i64> = (0..=255).collect();
    let indices: Vec<u8> = vec![255; 20];

    let result = unsafe { take_avx2(&values, &indices) };
    assert_eq!(&[255; 20], result.as_slice());
}

#[test]
fn last_valid_u16_index() {
    let values: Vec<i64> = (0..=65535).collect();
    let indices: Vec<u16> = vec![65535; 20];

    let result = unsafe { take_avx2(&values, &indices) };
    assert_eq!(&[65535; 20], result.as_slice());
}

#[test]
#[should_panic(expected = "take index out of bounds")]
fn invalid_index_only_in_simd_block() {
    let values = vec![10u32, 20, 30];
    let indices = vec![3u32, 0, 1, 2, 0, 1, 2, 0, 1];

    drop(unsafe { take_avx2(&values, &indices) });
}

#[test]
fn simd_array_u8x4() {
    let values: Vec<[u8; 4]> = (1u32..=200).map(u32::to_le_bytes).collect();
    let indices: Vec<u32> = (0..200).collect();

    let result = unsafe { take_avx2(&values, &indices) };
    assert_eq!(values.as_slice(), result.as_slice());
}

#[test]
fn scalar_fallback_u16() {
    let values: Vec<u16> = (1..=300).collect();
    let indices: Vec<u32> = (0..300).collect();

    let result = unsafe { take_avx2(&values, &indices) };
    assert_eq!(values.as_slice(), result.as_slice());
}

#[test]
fn scalar_fallback_array_u8x16() {
    let values: Vec<[u8; 16]> = (0u128..200).map(u128::to_le_bytes).collect();
    let indices: Vec<u32> = (0..200).collect();

    let result = unsafe { take_avx2(&values, &indices) };
    assert_eq!(values.as_slice(), result.as_slice());
}
