// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::arch::x86_64::_mm_loadu_si128;
use std::arch::x86_64::_mm_setzero_si128;
use std::arch::x86_64::_mm_shuffle_epi32;
use std::arch::x86_64::_mm_storeu_si128;
use std::arch::x86_64::_mm_unpacklo_epi64;
use std::arch::x86_64::_mm256_andnot_si256;
use std::arch::x86_64::_mm256_cmpgt_epi32;
use std::arch::x86_64::_mm256_cmpgt_epi64;
use std::arch::x86_64::_mm256_cvtepu8_epi32;
use std::arch::x86_64::_mm256_cvtepu8_epi64;
use std::arch::x86_64::_mm256_cvtepu16_epi32;
use std::arch::x86_64::_mm256_cvtepu16_epi64;
use std::arch::x86_64::_mm256_cvtepu32_epi64;
use std::arch::x86_64::_mm256_extracti128_si256;
use std::arch::x86_64::_mm256_loadu_si256;
use std::arch::x86_64::_mm256_mask_i32gather_epi32;
use std::arch::x86_64::_mm256_mask_i64gather_epi32;
use std::arch::x86_64::_mm256_mask_i64gather_epi64;
use std::arch::x86_64::_mm256_movemask_epi8;
use std::arch::x86_64::_mm256_set1_epi32;
use std::arch::x86_64::_mm256_set1_epi64x;
use std::arch::x86_64::_mm256_setzero_si256;
use std::arch::x86_64::_mm256_storeu_si256;
use std::convert::identity;

/// Moves one SIMD block of fixed-width values into an output buffer.
pub(super) trait GatherFn<Index, Lane> {
    /// The number of data elements written on each iteration.
    const WIDTH: usize;
    /// The number of indices read on each iteration.
    const STRIDE: usize = Self::WIDTH;

    /// Gather values from `src` into `dst`.
    ///
    /// # Safety
    ///
    /// `indices` must be readable for `STRIDE` elements. `src` must contain `max_idx + 1`
    /// elements, and `dst` must be writable for `WIDTH` elements.
    ///
    /// Returns whether every gathered index was at most `max_idx`.
    unsafe fn gather(
        indices: *const Index,
        max_idx: Index,
        src: *const Lane,
        dst: *mut Lane,
    ) -> bool;
}

/// AVX2 gather implementations for 32- and 64-bit value lanes.
pub(super) enum Avx2Gather {}

macro_rules! impl_gather {
    ($idx:ty, $({$value:ty => load: $load:ident, extend: $extend:ident, splat: $splat:ident, zero_vec: $zero_vec:ident, mask_indices: $mask_indices:ident, mask_cvt: |$mask_var:ident| $mask_cvt:block, gather: $masked_gather:ident, store: $store:ident, WIDTH = $WIDTH:literal, STRIDE = $STRIDE:literal }),+) => {
        $(
            impl_gather!(single; $idx, $value, load: $load, extend: $extend, splat: $splat, zero_vec: $zero_vec, mask_indices: $mask_indices, mask_cvt: |$mask_var| $mask_cvt, gather: $masked_gather, store: $store, WIDTH = $WIDTH, STRIDE = $STRIDE);
        )*
    };
    (single; $idx:ty, $value:ty, load: $load:ident, extend: $extend:ident, splat: $splat:ident, zero_vec: $zero_vec:ident, mask_indices: $mask_indices:ident, mask_cvt: |$mask_var:ident| $mask_cvt:block, gather: $masked_gather:ident, store: $store:ident, WIDTH = $WIDTH:literal, STRIDE = $STRIDE:literal) => {
        impl GatherFn<$idx, $value> for Avx2Gather {
            const WIDTH: usize = $WIDTH;
            const STRIDE: usize = $STRIDE;

            #[allow(unused_unsafe, clippy::cast_possible_truncation)]
            #[inline(always)]
            unsafe fn gather(
                indices: *const $idx,
                max_idx: $idx,
                src: *const $value,
                dst: *mut $value,
            ) -> bool {
                const {
                    assert!($WIDTH <= $STRIDE, "dst cannot advance by more than the stride");
                }

                const SCALE: i32 = std::mem::size_of::<$value>() as i32;

                let indices_vec = unsafe { $load(indices.cast()) };
                let indices_vec = unsafe { $extend(indices_vec) };
                let max_idx_vec = unsafe { $splat(max_idx as _) };
                let valid_mask = unsafe {
                    _mm256_andnot_si256(
                        $mask_indices(indices_vec, max_idx_vec),
                        $splat(-1),
                    )
                };
                let all_valid = unsafe { _mm256_movemask_epi8(valid_mask) == -1 };
                let valid_mask = {
                    let $mask_var = valid_mask;
                    $mask_cvt
                };
                let values_vec = unsafe {
                    $masked_gather::<SCALE>(
                        $zero_vec(),
                        src.cast(),
                        indices_vec,
                        valid_mask,
                    )
                };
                unsafe { $store(dst.cast(), values_vec) };
                all_valid
            }
        }
    };
}

impl_gather!(u8,
    { u32 =>
        load: _mm_loadu_si128,
        extend: _mm256_cvtepu8_epi32,
        splat: _mm256_set1_epi32,
        zero_vec: _mm256_setzero_si256,
        mask_indices: _mm256_cmpgt_epi32,
        mask_cvt: |x| { x },
        gather: _mm256_mask_i32gather_epi32,
        store: _mm256_storeu_si256,
        WIDTH = 8, STRIDE = 16
    },
    { u64 =>
        load: _mm_loadu_si128,
        extend: _mm256_cvtepu8_epi64,
        splat: _mm256_set1_epi64x,
        zero_vec: _mm256_setzero_si256,
        mask_indices: _mm256_cmpgt_epi64,
        mask_cvt: |x| { x },
        gather: _mm256_mask_i64gather_epi64,
        store: _mm256_storeu_si256,
        WIDTH = 4, STRIDE = 16
    }
);

impl_gather!(u16,
    { u32 =>
        load: _mm_loadu_si128,
        extend: _mm256_cvtepu16_epi32,
        splat: _mm256_set1_epi32,
        zero_vec: _mm256_setzero_si256,
        mask_indices: _mm256_cmpgt_epi32,
        mask_cvt: |x| { x },
        gather: _mm256_mask_i32gather_epi32,
        store: _mm256_storeu_si256,
        WIDTH = 8, STRIDE = 8
    },
    { u64 =>
        load: _mm_loadu_si128,
        extend: _mm256_cvtepu16_epi64,
        splat: _mm256_set1_epi64x,
        zero_vec: _mm256_setzero_si256,
        mask_indices: _mm256_cmpgt_epi64,
        mask_cvt: |x| { x },
        gather: _mm256_mask_i64gather_epi64,
        store: _mm256_storeu_si256,
        WIDTH = 4, STRIDE = 8
    }
);

impl_gather!(u32,
    { u32 =>
        load: _mm256_loadu_si256,
        extend: identity,
        splat: _mm256_set1_epi32,
        zero_vec: _mm256_setzero_si256,
        mask_indices: _mm256_cmpgt_epi32,
        mask_cvt: |x| { x },
        gather: _mm256_mask_i32gather_epi32,
        store: _mm256_storeu_si256,
        WIDTH = 8, STRIDE = 8
    },
    { u64 =>
        load: _mm_loadu_si128,
        extend: _mm256_cvtepu32_epi64,
        splat: _mm256_set1_epi64x,
        zero_vec: _mm256_setzero_si256,
        mask_indices: _mm256_cmpgt_epi64,
        mask_cvt: |x| { x },
        gather: _mm256_mask_i64gather_epi64,
        store: _mm256_storeu_si256,
        WIDTH = 4, STRIDE = 4
    }
);

impl_gather!(u64,
    { u32 =>
        load: _mm256_loadu_si256,
        extend: identity,
        splat: _mm256_set1_epi64x,
        zero_vec: _mm_setzero_si128,
        mask_indices: _mm256_cmpgt_epi64,
        mask_cvt: |mask| {
            unsafe {
                let lo_bits = _mm256_extracti128_si256::<0>(mask);
                let hi_bits = _mm256_extracti128_si256::<1>(mask);
                let lo_packed = _mm_shuffle_epi32::<0b01_01_01_01>(lo_bits);
                let hi_packed = _mm_shuffle_epi32::<0b01_01_01_01>(hi_bits);
                _mm_unpacklo_epi64(lo_packed, hi_packed)
            }
        },
        gather: _mm256_mask_i64gather_epi32,
        store: _mm_storeu_si128,
        WIDTH = 4, STRIDE = 4
    },
    { u64 =>
        load: _mm256_loadu_si256,
        extend: identity,
        splat: _mm256_set1_epi64x,
        zero_vec: _mm256_setzero_si256,
        mask_indices: _mm256_cmpgt_epi64,
        mask_cvt: |x| { x },
        gather: _mm256_mask_i64gather_epi64,
        store: _mm256_storeu_si256,
        WIDTH = 4, STRIDE = 4
    }
);
