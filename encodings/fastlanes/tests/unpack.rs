// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Tests for unpack_u32 (REBUILD Part 3 #8): the SVE tier must produce
// byte-identical output to the fastlanes BitPacking baseline at every
// bit width, and the public entry must handle partial trailing blocks.

use fastlanes::BitPacking;
use vortex_fastlanes::flatland::unpack::{
    unpack_block_u32_fastlanes, unpack_block_u32_sve, unpack_u32,
};

fn packed_words(width: u8, seed: u64) -> Vec<u32> {
    // Deterministic nonzero words: every bit pattern exercised over the sweep.
    (0..32 * width as usize)
        .map(|i| {
            (i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(seed)
                .rotate_left(17) as u32
                | 0x8000_0001
        })
        .collect()
}

fn baseline_block(packed: &[u32], width: u8) -> [u32; 1024] {
    let mut out = [0u32; 1024];
    // SAFETY: full block at the contract width.
    unsafe { BitPacking::unchecked_unpack(width as usize, packed, &mut out) };
    out
}

#[test]
fn sve_tier_matches_fastlanes_baseline_all_widths() {
    for width in 1u8..=32 {
        let packed = packed_words(width, 0xDEADBEEF);
        let mut out = [0u32; 1024];
        // SAFETY: full block at the contract width; this host has SVE.
        unsafe { unpack_block_u32_sve(&packed, width as usize, &mut out) };
        assert_eq!(out, baseline_block(&packed, width), "width {width}");
    }
}

#[test]
fn unpack_u32_partial_tail() {
    let width = 5u8;
    let packed: Vec<u32> = vec![0xDEADBEEF; 2 * 32 * width as usize];
    let mut out = vec![0u32; 1500]; // 1 full block + 476 tail
    unpack_u32(&packed, width, &mut out);
    assert_eq!(&out[..1024], &baseline_block(&packed, width));
    // Tail re-decodes the second block's prefix.
    let second = &packed[32 * width as usize..];
    assert_eq!(&out[1024..1500], &baseline_block(second, width)[..476]);
}

#[test]
fn unpack_u32_width_sweep_entry() {
    // The dispatched entry (SVE probe → fastlanes) at a few widths, single
    // and multi-block, against the baseline.
    for width in [1u8, 7, 8, 17, 31, 32] {
        // Two blocks of packed words (the entry consumes block-by-block).
        let one_block = packed_words(width, 7);
        let packed: Vec<u32> = one_block.iter().chain(&one_block).copied().collect();
        let mut out = vec![0u32; 2048]; // 2 full blocks
        unpack_u32(&packed, width, &mut out);
        let expect = baseline_block(&one_block, width);
        assert_eq!(&out[..1024], &expect, "width {width} block 0");
        assert_eq!(&out[1024..], &expect, "width {width} block 1");
    }
}

#[test]
fn unpack_u32_zero_width_is_all_zero() {
    let packed: Vec<u32> = vec![];
    let mut out = vec![9u32; 1024];
    unpack_u32(&packed, 0, &mut out);
    assert!(out.iter().all(|&v| v == 0));
}
