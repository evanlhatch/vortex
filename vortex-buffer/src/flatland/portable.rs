// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Portable SIMD tiers for the flatland elementwise stages (REBUILD Part 8,
//! portable-default policy).
//!
//! Uses `std::simd` (`portable_simd`, nightly) with a fixed lane width so
//! ONE implementation lowers to the best fixed-width ISA on every target
//! (AVX2/AVX-512 on x86, NEON on aarch64) — no per-ISA clones. This is the
//! DEFAULT tier for every elementwise op on every machine: best performance
//! by default with zero per-ISA dispatch. [`super::sve`] keeps only the
//! operations `std::simd` cannot express (varlen gather/scatter/compact).
//!
//! The kernels here are the elementwise stages of the fused loop: add-const,
//! inequality lanes, bitwise and/or/xor/not. (The varlen unpack stage is
//! deliberately NOT here — cross-lane bit extraction is where fixed-lane
//! portable SIMD is weakest.)

use std::simd::cmp::SimdPartialEq;
use std::simd::Simd;
use std::simd::Mask;
use std::simd::Select;

use crate::CpuKernel;

/// Lane width for the portable tier. 256-bit with u32 lanes.
const LANES: usize = 8;

/// Kernel shape shared by the binary elementwise stages.
/// Kernel shape: two u32 slices in, one out.
type U32BinaryKernel = fn(&[u32], &[u32], &mut [u32]);

/// Unary u32 kernel shape (Not).
type U32UnaryKernel = fn(&[u32], &mut [u32]);

/// Constant-add kernel shape.
type U32AddConstKernel = fn(&[u32], u32, &mut [u32]);

/// `out[i] = if lhs[i] != rhs[i] { 1 } else { 0 }`.
pub fn neq_lanes_u32(lhs: &[u32], rhs: &[u32], out: &mut [u32]) {
    static KERNEL: CpuKernel<U32BinaryKernel> =
        CpuKernel::new(|| neq_u32_portable);
    KERNEL.get()(lhs, rhs, out)
}

pub(crate) fn neq_u32_portable(lhs: &[u32], rhs: &[u32], out: &mut [u32]) {
    debug_assert_eq!(lhs.len(), rhs.len());
    let one = Simd::<u32, LANES>::splat(1);
    let zero = Simd::<u32, LANES>::splat(0);
    let mut idx = 0;
    while idx + LANES <= lhs.len() {
        let av = Simd::<u32, LANES>::from_slice(&lhs[idx..idx + LANES]);
        let bv = Simd::<u32, LANES>::from_slice(&rhs[idx..idx + LANES]);
        av.simd_ne(bv)
            .select(one, zero)
            .copy_to_slice(&mut out[idx..idx + LANES]);
        idx += LANES;
    }
    // Masked tail: one lane op covers the remainder (no scalar loop).
    while idx < lhs.len() {
        let valid = lane_valid(idx, lhs.len());
        let lhs_lanes = Simd::<u32, LANES>::load_select_or_default(&lhs[idx..], valid);
        let rhs_lanes = Simd::<u32, LANES>::load_select_or_default(&rhs[idx..], valid);
        (lhs_lanes.simd_ne(rhs_lanes).select(Simd::splat(1), Simd::splat(0)))
            .store_select(&mut out[idx..], valid);
        idx += LANES;
    }
}

/// `out[i] = a[i] & b[i]`.
pub fn and_u32(lhs: &[u32], rhs: &[u32], out: &mut [u32]) {
    static KERNEL: CpuKernel<U32BinaryKernel> =
        CpuKernel::new(|| and_u32_portable);
    KERNEL.get()(lhs, rhs, out)
}

pub(crate) fn and_u32_portable(lhs: &[u32], rhs: &[u32], out: &mut [u32]) {
    binary_loop(lhs, rhs, out, |x, y| x & y)
}

/// `out[i] = a[i] | b[i]`.
pub fn or_u32(lhs: &[u32], rhs: &[u32], out: &mut [u32]) {
    static KERNEL: CpuKernel<U32BinaryKernel> =
        CpuKernel::new(|| or_u32_portable);
    KERNEL.get()(lhs, rhs, out)
}

pub(crate) fn or_u32_portable(lhs: &[u32], rhs: &[u32], out: &mut [u32]) {
    binary_loop(lhs, rhs, out, |x, y| x | y)
}

/// `out[i] = a[i] ^ b[i]`.
pub fn xor_u32(lhs: &[u32], rhs: &[u32], out: &mut [u32]) {
    static KERNEL: CpuKernel<U32BinaryKernel> =
        CpuKernel::new(|| xor_u32_portable);
    KERNEL.get()(lhs, rhs, out)
}

pub(crate) fn xor_u32_portable(lhs: &[u32], rhs: &[u32], out: &mut [u32]) {
    binary_loop(lhs, rhs, out, |x, y| x ^ y)
}

/// Generic binary SIMD loop: `out[i] = f(lhs[i], rhs[i])` over lane chunks + tail.
#[inline]
fn binary_loop(
    lhs: &[u32],
    rhs: &[u32],
    out: &mut [u32],
    f: impl Fn(Simd<u32, LANES>, Simd<u32, LANES>) -> Simd<u32, LANES>,
) {
    debug_assert_eq!(lhs.len(), rhs.len());
    debug_assert_eq!(lhs.len(), out.len());
    let len = lhs.len();
    let mut idx = 0;
    while idx + LANES <= len {
        let lhs_lanes = Simd::<u32, LANES>::from_slice(&lhs[idx..idx + LANES]);
        let rhs_lanes = Simd::<u32, LANES>::from_slice(&rhs[idx..idx + LANES]);
        f(lhs_lanes, rhs_lanes).copy_to_slice(&mut out[idx..idx + LANES]);
        idx += LANES;
    }
    // Masked tail: one lane op for the remainder (no scalar loop).
    while idx < len {
        let valid = lane_valid(idx, len);
        let lhs_lanes = Simd::<u32, LANES>::load_select_or_default(&lhs[idx..], valid);
        let rhs_lanes = Simd::<u32, LANES>::load_select_or_default(&rhs[idx..], valid);
        f(lhs_lanes, rhs_lanes).store_select(&mut out[idx..], valid);
        idx += LANES;
    }
}

/// `out[i] = !a[i]`.
pub fn not_u32(lhs: &[u32], out: &mut [u32]) {
    static KERNEL: CpuKernel<U32UnaryKernel> = CpuKernel::new(|| not_u32_portable);
    KERNEL.get()(lhs, out)
}

pub(crate) fn not_u32_portable(values: &[u32], out: &mut [u32]) {
    let len = values.len();
    let mut idx = 0;
    while idx + LANES <= len {
        let lane = Simd::<u32, LANES>::from_slice(&values[idx..idx + LANES]);
        (!lane).copy_to_slice(&mut out[idx..idx + LANES]);
        idx += LANES;
    }
    // Masked tail: one lane op for the remainder (no scalar loop).
    while idx < len {
        let valid = lane_valid(idx, len);
        let lanes = Simd::<u32, LANES>::load_select_or_default(&values[idx..], valid);
        (!lanes).store_select(&mut out[idx..], valid);
        idx += LANES;
    }
}

/// `out[i] = a[i] + b[i]` (wrapping).
pub fn add_u32(lhs: &[u32], rhs: &[u32], out: &mut [u32]) {
    static KERNEL: CpuKernel<U32BinaryKernel> =
        CpuKernel::new(|| add_u32_portable);
    KERNEL.get()(lhs, rhs, out)
}

pub(crate) fn add_u32_portable(lhs: &[u32], rhs: &[u32], out: &mut [u32]) {
    binary_loop(lhs, rhs, out, |x, y| x + y) // Simd int ops wrap, matching engine modular arithmetic
}

/// `out[i] = a[i] - b[i]` (wrapping).
pub fn sub_u32(lhs: &[u32], rhs: &[u32], out: &mut [u32]) {
    static KERNEL: CpuKernel<U32BinaryKernel> =
        CpuKernel::new(|| sub_u32_portable);
    KERNEL.get()(lhs, rhs, out)
}

pub(crate) fn sub_u32_portable(lhs: &[u32], rhs: &[u32], out: &mut [u32]) {
    binary_loop(lhs, rhs, out, |x, y| x - y)
}

/// `out[i] = min(a[i], b[i])`.
pub fn min_u32(lhs: &[u32], rhs: &[u32], out: &mut [u32]) {
    static KERNEL: CpuKernel<U32BinaryKernel> =
        CpuKernel::new(|| min_u32_portable);
    KERNEL.get()(lhs, rhs, out)
}

pub(crate) fn min_u32_portable(lhs: &[u32], rhs: &[u32], out: &mut [u32]) {
    use std::simd::cmp::SimdPartialOrd;
    binary_loop(lhs, rhs, out, |x, y| y.simd_lt(x).select(y, x))
    // Simd::min/max are Ord-lexicographic on integers — lane-wise min needs select.
}

/// `out[i] = a[i].max(b[i])`.
pub fn max_u32(lhs: &[u32], rhs: &[u32], out: &mut [u32]) {
    static KERNEL: CpuKernel<U32BinaryKernel> =
        CpuKernel::new(|| max_u32_portable);
    KERNEL.get()(lhs, rhs, out)
}

pub(crate) fn max_u32_portable(lhs: &[u32], rhs: &[u32], out: &mut [u32]) {
    use std::simd::cmp::SimdPartialOrd;
    binary_loop(lhs, rhs, out, |x, y| x.simd_gt(y).select(x, y))
}

/// `out[i] = a[i].clamp(lo, hi)` (lo <= hi contract; portable is the default tier).
pub fn clamp_const_u32(values: &[u32], lo: u32, hi: u32, out: &mut [u32]) {
    use std::simd::cmp::SimdPartialOrd;
    let lv = Simd::<u32, LANES>::splat(lo);
    let hv = Simd::<u32, LANES>::splat(hi);
    let len = values.len();
    let mut idx = 0;
    while idx + LANES <= len {
        let lanes = Simd::<u32, LANES>::from_slice(&values[idx..idx + LANES]);
        // Lane-wise max(lo) then min(hi) via mask select — Simd::min/max are
        // Ord-lexicographic on integers, NOT lane-wise.
        let raised = lanes.simd_lt(lv).select(lv, lanes);
        raised.simd_gt(hv).select(hv, raised).copy_to_slice(&mut out[idx..idx + LANES]);
        idx += LANES;
    }
    while idx < len {
        out[idx] = values[idx].clamp(lo, hi);
        idx += 1;
    }
}

/// `out[i] = a[i] + constant` (wrapping).
pub fn add_const_u32(values: &[u32], constant: u32, out: &mut [u32]) {
    static KERNEL: CpuKernel<U32AddConstKernel> =
        CpuKernel::new(|| add_const_portable);
    KERNEL.get()(values, constant, out)
}

pub(crate) fn add_const_portable(values: &[u32], constant: u32, out: &mut [u32]) {
    let constant_lanes = Simd::<u32, LANES>::splat(constant);
    let mut idx = 0;
    while idx + LANES <= values.len() {
        let lanes = Simd::<u32, LANES>::from_slice(&values[idx..idx + LANES]);
        (lanes + constant_lanes).copy_to_slice(&mut out[idx..idx + LANES]);
        idx += LANES;
    }
    // Masked tail: one lane op for the remainder (no scalar loop).
    while idx < values.len() {
        let valid = lane_valid(idx, values.len());
        let lanes = Simd::<u32, LANES>::load_select_or_default(&values[idx..], valid);
        (lanes + constant_lanes).store_select(&mut out[idx..], valid);
        idx += LANES;
    }
}



// =============================================================================
// Fused compare→index-extract (portable varlen compaction, REBUILD Part 3 #5)
// =============================================================================

/// Lane-positions of set bits for every 8-lane mask: entry `m` lists the
/// source lanes selected by bitmask `m` (low to high). Padding lanes repeat
/// the last kept lane — callers only store the first `popcount(mask)` lanes.
const COMPACT_TABLE: [[usize; LANES]; 256] = {
    let mut table = [[0usize; LANES]; 256];
    let mut mask = 0usize;
    while mask < 256 {
        let mut kept = 0usize;
        let mut bit = 0usize;
        while bit < 8 {
            if mask & (1 << bit) != 0 {
                table[mask][kept] = bit;
                kept += 1;
            }
            bit += 1;
        }
        mask += 1;
    }
    table
};

/// Fused inequality→packed indices: `out` gains the index of every row where
/// `a[i] != b[i]`, packed contiguously; returns the total written. The diff
/// hot path's compare+extract in one pass — no lanes intermediate, no scalar
/// scan. Portable tier: SIMD compare → `Mask::to_bitmask` → per-mask lane
/// table → `gather_or` (register compaction; VPERMD-class on x86, per-lane
/// lowering elsewhere — zero per-ISA intrinsics).
pub fn neq_indices_u32(a: &[u32], b: &[u32], out: &mut Vec<u32>) -> usize {
    debug_assert_eq!(a.len(), b.len());
    out.clear();
    let len = a.len();
    let mut idx = 0;
    let mut written = 0usize;
    // Full chunks: compare → bitmask → gather-table compaction.
    while idx + LANES <= len {
        // u32 row-index convention (flatland: <= 2^32 rows per column).
        #[allow(clippy::cast_possible_truncation, reason = "u32 row-index convention")]
        let iota_arr: [u32; LANES] = std::array::from_fn(|k| (idx + k) as u32);
        let changed = Simd::<u32, LANES>::from_slice(&a[idx..])
            .simd_ne(Simd::<u32, LANES>::from_slice(&b[idx..]));
        // LANES = 8: the bitmask fits a byte by construction.
        #[allow(clippy::cast_possible_truncation, reason = "8 lanes fit u8")]
        let bits = changed.to_bitmask() as u8;
        if bits != 0 {
            let count = bits.count_ones() as usize;
            let compacted: Simd<u32, LANES> = Simd::gather_or(
                &iota_arr,
                Simd::from_array(COMPACT_TABLE[bits as usize]),
                Simd::splat(0),
            );
            out.resize(written + count, 0);
            out[written..written + count].copy_from_slice(&compacted.as_array()[..count]);
            written += count;
        }
        idx += LANES;
    }
    // Scalar tail.
    while idx < len {
        #[allow(clippy::cast_possible_truncation, reason = "u32 row-index convention")]
        if a[idx] != b[idx] {
            written += 1;
            out.push(idx as u32);
        }
        idx += 1;
    }
    written
}

// =============================================================================
// Portable gather/scatter (Part 3 #4 upgrade — replaces the scalar fallbacks)
// =============================================================================

/// Portable gather tier: `out[i] = src[keys[i]]`, out-of-range keys read 0.
/// `gather_select` bounds-masks per lane — safe by construction. Lowers to
/// per-lane loads (AVX-512 vgather where present) on every platform.
pub fn gather_u32_portable(src: &[u32], keys: &[u32], out: &mut [u32]) {
    let mut idx = 0;
    while idx < keys.len() {
        let valid: Mask<i32, LANES> = lane_valid(idx, keys.len());
        let key_lanes: Simd<u32, LANES> = Simd::load_select_or_default(&keys[idx..], valid);
        let gathered: Simd<u32, LANES> = Simd::gather_select(
            src,
            Mask::splat(true),
            Simd::from_array(key_lanes.to_array().map(|k| k as usize)),
            Simd::splat(0),
        );
        gathered.store_select(&mut out[idx..], valid);
        idx += LANES;
    }
}

/// Portable scatter: `out[keys[i]] = vals[i]`, out-of-range keys dropped.
/// `scatter_select` suppresses OOB writes — the scalar `get_mut` semantics.
pub fn scatter_u32_portable(keys: &[u32], vals: &[u32], out: &mut [u32]) {
    let n = keys.len().min(out.len());
    let mut idx = 0;
    while idx < n {
        let valid = lane_valid(idx, n);
        let key_lanes: Simd<u32, LANES> = Simd::load_select_or_default(&keys[idx..], valid);
        let val_lanes: Simd<u32, LANES> = Simd::load_select_or_default(&vals[idx..], valid);
        val_lanes.scatter_select(
            out,
            Mask::from_bitmask(valid.to_bitmask()),
            Simd::from_array(key_lanes.to_array().map(|k| k as usize)),
        );
        idx += LANES;
    }
}

/// LANES-wide chunk validity mask.
#[inline]
fn lane_valid(idx: usize, len: usize) -> Mask<i32, LANES> {
    Mask::from_array(std::array::from_fn(|k| idx + k < len))
}

// =============================================================================
// Fused multiply-add-const (the affine Primitive mul path, native lanes)
// =============================================================================

/// `out[i] = values[i] * constant + addend` (wrapping), native-lane SIMD.
/// The affine kernel's mul-path body — replaces the scalar wrap loop, kills
/// the f64 interchange (u64 columns keep exactness above 2^53).
pub fn mul_add_const_u32(values: &[u32], constant: u32, addend: u32, out: &mut [u32]) {
    let cv = Simd::<u32, LANES>::splat(constant);
    let av = Simd::<u32, LANES>::splat(addend);
    let mut idx = 0;
    while idx + LANES <= values.len() {
        let lanes = Simd::<u32, LANES>::from_slice(&values[idx..]);
        (lanes * cv + av).copy_to_slice(&mut out[idx..idx + LANES]);
        idx += LANES;
    }
    // Masked tail: one lane op for the remainder (no scalar loop).
    while idx < values.len() {
        let valid = lane_valid(idx, values.len());
        let lanes = Simd::<u32, LANES>::load_select_or_default(&values[idx..], valid);
        (lanes * cv + av).store_select(&mut out[idx..], valid);
        idx += LANES;
    }
}
