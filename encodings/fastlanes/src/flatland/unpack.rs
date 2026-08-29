// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Runtime-width FastLanes-layout u32 unpack with an SVE tier (flatland
//! REBUILD Part 3 #8 — the register-level decode primitive).
//!
//! `unpack_u32` decodes 1024-value FastLanes blocks at a runtime bit width
//! (1–32, the flatland convention: u32 keys, packed widths 1–32). Dispatch
//! per [`CpuKernel`]: SVE (aarch64, runtime-probed) → fastlanes
//! [`BitPacking::unchecked_unpack`] (the portable SIMD baseline — never a
//! hand-scalar fallback, per the portable-default policy).
//!
//! The SVE tier vectorizes over the FastLanes *lane* dimension, where the
//! layout is dense and contiguous:
//! - value bits for (row, lane) live at `packed[32 * (row*W)/32 + lane]`
//!   (words are lane-interleaved, so a row's source words are contiguous),
//! - decoded values land at `FL_ORDER[row/8]*16 + (row%8)*128 + lane`
//!   (also lane-contiguous — stores need no scatter).
//!
//! That makes the whole kernel `svld1`/shift-mask/`svst1` with no gathers,
//! at hardware vector length — the honest SVE donation; the item-6 bench
//! verifies it against the fastlanes baseline on the SVE2 host.

use vortex_buffer::CpuKernel;

use fastlanes::BitPacking;

const FL_ORDER: [usize; 8] = [0, 4, 2, 6, 1, 5, 3, 7];
const ROWS: usize = 32;
const LANES: usize = 32;
const BLOCK: usize = 1024;

/// Unpack `out.len()` values from FastLanes-packed u32 words.
///
/// `packed` must hold `ceil(out.len()/1024) * 32 * bit_width / 4` words
/// (whole 1024-value blocks; a partial final block is decoded through stack
/// scratch and truncated). `bit_width` must be 0..=32.
pub fn unpack_u32(packed: &[u32], bit_width: u8, out: &mut [u32]) {
    assert!(bit_width <= 32, "bit_width {bit_width} exceeds u32");
    let words_per_block = 32 * bit_width as usize;
    let n_blocks = out.len().div_ceil(BLOCK);
    debug_assert!(
        packed.len() >= n_blocks * words_per_block,
        "packed length {} too small for {} blocks at width {}",
        packed.len(),
        n_blocks,
        bit_width
    );

    static KERNEL: CpuKernel<unsafe fn(&[u32], usize, &mut [u32; BLOCK])> = CpuKernel::new(|| {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("sve") {
                return unpack_block_u32_sve;
            }
        }
        unpack_block_u32_fastlanes
    });

    let mut scratch = [0u32; BLOCK];
    let full_blocks = out.len() / BLOCK;
    for block in 0..full_blocks {
        let words = &packed[block * words_per_block..][..words_per_block];
        // SAFETY: the selector probed the required feature; both tiers write
        // exactly BLOCK values from words_per_block packed words.
        let out_block: &mut [u32; BLOCK] =
            (&mut out[block * BLOCK..][..BLOCK]).try_into().unwrap();
        unsafe { (KERNEL.get())(words, bit_width as usize, out_block) };
    }
    let tail = out.len() % BLOCK;
    if tail > 0 {
        let words = &packed[full_blocks * words_per_block..][..words_per_block];
        // Partial trailing block: decode into scratch, copy the tail.
        // SAFETY: as above; scratch is exactly BLOCK long.
        unsafe { (KERNEL.get())(words, bit_width as usize, &mut scratch) };
        out[full_blocks * BLOCK..].copy_from_slice(&scratch[..tail]);
    }
}

/// fastlanes baseline tier: the crate's own SIMD unpack at runtime width.
/// `pub` so the tier-parity tests/benches can pin both sides directly.
pub fn unpack_block_u32_fastlanes(packed: &[u32], width: usize, out: &mut [u32; BLOCK]) {
    // SAFETY: packed.len() == 32*width (checked by the caller's slicing) and
    // out is exactly 1024 — `unchecked_unpack`'s contract.
    unsafe { BitPacking::unchecked_unpack(width, packed, out) }
}

/// SVE tier: lane-vectorized FastLanes unpack, no gathers. `pub unsafe` for
/// the tier-parity tests/benches; callers must uphold the block contract.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve")]
pub unsafe fn unpack_block_u32_sve(packed: &[u32], width: usize, out: &mut [u32; BLOCK]) {
    use std::arch::aarch64::{
        svand_n_u32_x, svcntw, svld1_u32, svlsr_n_u32_x, svlsl_n_u32_x, svorr_u32_x, svst1_u32,
        svwhilelt_b32_u32,
    };
    unsafe {
        debug_assert_eq!(packed.len(), 32 * width);
        if width == 0 {
            // W=0: every value is 0 — fill directly.
            out.fill(0);
            return;
        }
        let vl = svcntw() as usize;
        // Lane-vector loop: the SVE register holds `vl` consecutive lanes.
        // All addressing below is per-row; lanes are the vector dimension.
        for row in 0..ROWS {
            let curr_word = (row * width) / 32;
            let next_word = ((row + 1) * width) / 32;
            let shift = (row * width) % 32;
            let base = FL_ORDER[row / 8] * 16 + (row % 8) * 128;

            if width == 32 {
                // Transposed copy: out[index(row, lane)] = packed[32*row + lane].
                for lc in (0..LANES).step_by(vl) {
                    let pg = svwhilelt_b32_u32(lc as u32, LANES as u32);
                    let v = svld1_u32(pg, packed.as_ptr().add(row * LANES + lc));
                    svst1_u32(pg, out.as_mut_ptr().add(base + lc), v);
                }
                continue;
            }

            if next_word > curr_word {
                // Row's W bits straddle a word boundary:
                // low bits (32-shift) from curr word, rem bits from next.
                // shift + width > 32 here, so 32 - shift == width - rem.
                let rem = ((row + 1) * width) % 32;
                let cur_bits = 32 - shift;
                let mask_rem = (1u32 << rem) - 1; // rem == 0 → no next-word bits
                for lc in (0..LANES).step_by(vl) {
                    let pg = svwhilelt_b32_u32(lc as u32, LANES as u32);
                    let src1 = svld1_u32(pg, packed.as_ptr().add(curr_word * LANES + lc));
                    // Low cur_bits are already exact (no mask needed):
                    // cur_bits == 32 - shift, so bits above them are gone.
                    let x = if shift > 0 {
                        svlsr_n_u32_x(pg, src1, shift as u32)
                    } else {
                        src1
                    };
                    let src2 = svld1_u32(pg, packed.as_ptr().add(next_word * LANES + lc));
                    let low = svand_n_u32_x(pg, src2, mask_rem);
                    let high = svlsl_n_u32_x(pg, low, cur_bits as u32);
                    let x = svorr_u32_x(pg, x, high);
                    // Bits above width: cur_bits + rem == width, so none.
                    svst1_u32(pg, out.as_mut_ptr().add(base + lc), x);
                }
            } else {
                // Row's W bits sit inside one word: shift up, mask to width.
                let mask_w = (1u32 << width) - 1;
                for lc in (0..LANES).step_by(vl) {
                    let pg = svwhilelt_b32_u32(lc as u32, LANES as u32);
                    let src1 = svld1_u32(pg, packed.as_ptr().add(curr_word * LANES + lc));
                    let shifted = if shift > 0 {
                        svlsr_n_u32_x(pg, src1, shift as u32)
                    } else {
                        src1
                    };
                    let x = svand_n_u32_x(pg, shifted, mask_w);
                    svst1_u32(pg, out.as_mut_ptr().add(base + lc), x);
                }
            }
        }
    }
}


