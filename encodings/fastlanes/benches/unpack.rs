// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Bench for unpack_u32 (REBUILD Part 3 #8 + Part 8 obligation): the SVE
// lane-vector tier against the fastlanes BitPacking baseline, per width.
// This is the honest arbiter for the SVE donation — run on the SVE2 host:
//
//     cargo bench -p vortex-fastlanes --bench unpack


#![allow(clippy::cast_possible_truncation, reason = "flatland u32-key convention in tests/benches")]
#![allow(clippy::redundant_clone, reason = "test fixtures; clarity over micro-optimization")]

use std::hint::black_box;

use divan::Bencher;
use vortex_fastlanes::flatland::unpack::{
    unpack_block_u32_fastlanes, unpack_block_u32_sve, unpack_u32,
};

/// 8 blocks = 8192 values (working set beyond L1; realistic column chunk).
const BLOCKS: usize = 8;
const BLOCK: usize = 1024;

fn gen_packed(width: u8) -> Vec<u32> {
    (0..BLOCKS * 32 * width as usize)
        .map(|i| {
            (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .rotate_left(17) as u32
                | 0x8000_0001
        })
        .collect()
}

fn gen_packed_direct(width: u8) -> Vec<u32> {
    // Per-block words for the direct tier comparisons.
    (0..32 * width as usize)
        .map(|i| {
            (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .rotate_left(17) as u32
                | 0x8000_0001
        })
        .collect()
}

fn main() {
    divan::main();
}

macro_rules! width_benches {
    ($mod_name:ident, $width:expr) => {
        mod $mod_name {
            use super::*;

            #[divan::bench]
            fn sve_tier(b: Bencher) {
                let packed = gen_packed_direct($width);
                b.bench(|| {
                    let mut out = [0u32; BLOCK];
                    // SAFETY: full block at the contract width.
                    unsafe {
                        unpack_block_u32_sve(black_box(&packed), $width as usize, &mut out)
                    };
                    black_box(&out);
                });
            }

            #[divan::bench]
            fn fastlanes_baseline(b: Bencher) {
                let packed = gen_packed_direct($width);
                b.bench(|| {
                    let mut out = [0u32; BLOCK];
                    unpack_block_u32_fastlanes(black_box(&packed), $width as usize, &mut out);
                    black_box(&out);
                });
            }

            #[divan::bench]
            fn dispatched_multi_block(b: Bencher) {
                let packed = gen_packed($width);
                b.bench(|| {
                    let mut out = vec![0u32; BLOCKS * BLOCK];
                    unpack_u32(black_box(&packed), $width, black_box(&mut out));
                    black_box(&out);
                });
            }
        }
    };
}

width_benches!(w01, 1);
width_benches!(w08, 8);
width_benches!(w16, 16);
width_benches!(w32, 32);
