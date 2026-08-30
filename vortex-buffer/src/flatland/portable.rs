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

use std::simd::Simd;
use std::simd::cmp::SimdPartialEq;
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
    let len = lhs.len();
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
    while idx < lhs.len() {
        out[idx] = if lhs[idx] != rhs[idx] { 1 } else { 0 };
        idx += 1;
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
    while idx < len {
        out[idx] = f(
            Simd::<u32, LANES>::splat(lhs[idx]),
            Simd::<u32, LANES>::splat(rhs[idx]),
        )[0];
        idx += 1;
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
    while idx < values.len() {
        out[idx] = !values[idx];
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
    let len = values.len();
    let mut idx = 0;
    while idx + LANES <= values.len() {
        let lanes = Simd::<u32, LANES>::from_slice(&values[idx..idx + LANES]);
        (lanes + constant_lanes).copy_to_slice(&mut out[idx..idx + LANES]);
        idx += LANES;
    }
    while idx < values.len() {
        out[idx] = values[idx].wrapping_add(constant);
        idx += 1;
    }
}
