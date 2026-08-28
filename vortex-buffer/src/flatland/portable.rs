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
use std::simd::Select;
use std::simd::cmp::SimdPartialEq;

use crate::CpuKernel;

/// Lane width for the portable tier. 256-bit with u32 lanes.
const LANES: usize = 8;

/// `out[i] = if a[i] != b[i] { 1 } else { 0 }`.
pub fn neq_lanes_u32(a: &[u32], b: &[u32], out: &mut [u32]) {
    static KERNEL: CpuKernel<fn(&[u32], &[u32], &mut [u32])> =
        CpuKernel::new(|| neq_u32_portable);
    KERNEL.get()(a, b, out)
}

pub(crate) fn neq_u32_portable(a: &[u32], b: &[u32], out: &mut [u32]) {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut i = 0;
    let one = Simd::<u32, LANES>::splat(1);
    let zero = Simd::<u32, LANES>::splat(0);
    while i + LANES <= n {
        let av = Simd::<u32, LANES>::from_slice(&a[i..i + LANES]);
        let bv = Simd::<u32, LANES>::from_slice(&b[i..i + LANES]);
        av.simd_ne(bv).select(one, zero).copy_to_slice(&mut out[i..i + LANES]);
        i += LANES;
    }
    while i < n {
        out[i] = if a[i] != b[i] { 1 } else { 0 };
        i += 1;
    }
}

/// `out[i] = a[i] & b[i]`.
pub fn and_u32(a: &[u32], b: &[u32], out: &mut [u32]) {
    static KERNEL: CpuKernel<fn(&[u32], &[u32], &mut [u32])> =
        CpuKernel::new(|| and_u32_portable);
    KERNEL.get()(a, b, out)
}

pub(crate) fn and_u32_portable(a: &[u32], b: &[u32], out: &mut [u32]) {
    binary_loop(a, b, out, |x, y| x & y)
}

/// `out[i] = a[i] | b[i]`.
pub fn or_u32(a: &[u32], b: &[u32], out: &mut [u32]) {
    static KERNEL: CpuKernel<fn(&[u32], &[u32], &mut [u32])> =
        CpuKernel::new(|| or_u32_portable);
    KERNEL.get()(a, b, out)
}

pub(crate) fn or_u32_portable(a: &[u32], b: &[u32], out: &mut [u32]) {
    binary_loop(a, b, out, |x, y| x | y)
}

/// `out[i] = a[i] ^ b[i]`.
pub fn xor_u32(a: &[u32], b: &[u32], out: &mut [u32]) {
    static KERNEL: CpuKernel<fn(&[u32], &[u32], &mut [u32])> =
        CpuKernel::new(|| xor_u32_portable);
    KERNEL.get()(a, b, out)
}

pub(crate) fn xor_u32_portable(a: &[u32], b: &[u32], out: &mut [u32]) {
    binary_loop(a, b, out, |x, y| x ^ y)
}

/// `out[i] = !a[i]`.
pub fn not_u32(a: &[u32], out: &mut [u32]) {
    static KERNEL: CpuKernel<fn(&[u32], &mut [u32])> = CpuKernel::new(|| not_u32_portable);
    KERNEL.get()(a, out)
}

pub(crate) fn not_u32_portable(a: &[u32], out: &mut [u32]) {
    let n = a.len();
    let mut i = 0;
    while i + LANES <= n {
        let av = Simd::<u32, LANES>::from_slice(&a[i..i + LANES]);
        (!av).copy_to_slice(&mut out[i..i + LANES]);
        i += LANES;
    }
    while i < n {
        out[i] = !a[i];
        i += 1;
    }
}

/// `out[i] = a[i] + c` (wrapping).
pub fn add_const_u32(a: &[u32], c: u32, out: &mut [u32]) {
    static KERNEL: CpuKernel<fn(&[u32], u32, &mut [u32])> =
        CpuKernel::new(|| add_const_portable);
    KERNEL.get()(a, c, out)
}

pub(crate) fn add_const_portable(a: &[u32], c: u32, out: &mut [u32]) {
    let n = a.len();
    let mut i = 0;
    let cv = Simd::<u32, LANES>::splat(c);
    while i + LANES <= n {
        let av = Simd::<u32, LANES>::from_slice(&a[i..i + LANES]);
        (av + cv).copy_to_slice(&mut out[i..i + LANES]);
        i += LANES;
    }
    while i < n {
        out[i] = a[i].wrapping_add(c);
        i += 1;
    }
}

/// Generic binary SIMD loop: `out[i] = f(a[i], b[i])` over lane chunks + tail.
#[inline]
fn binary_loop(a: &[u32], b: &[u32], out: &mut [u32], f: impl Fn(Simd<u32, LANES>, Simd<u32, LANES>) -> Simd<u32, LANES>) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let n = a.len();
    let mut i = 0;
    while i + LANES <= n {
        let av = Simd::<u32, LANES>::from_slice(&a[i..i + LANES]);
        let bv = Simd::<u32, LANES>::from_slice(&b[i..i + LANES]);
        f(av, bv).copy_to_slice(&mut out[i..i + LANES]);
        i += LANES;
    }
    while i < n {
        out[i] = f(Simd::<u32, LANES>::splat(a[i]), Simd::<u32, LANES>::splat(b[i]))[0];
        i += 1;
    }
}
