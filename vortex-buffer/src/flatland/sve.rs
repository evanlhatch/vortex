// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalable SVE kernels (flatland REBUILD Part 3 #4).
//!
//! Predicated, vector-length-scalable loops for the flatland hot verbs:
//! gather (indexed read), scatter (indexed write), inequality lanes
//! (the `diff` primitive's compare, Part 3 #5) and const-add lifts (Part
//! 13.5). One code path scales to the hardware vector length via
//! `svwhilelt`/`svcntw` — no fixed-width clones.
//!
//! Dispatch chain per [`CpuKernel`]: SVE → NEON → scalar. NEON (asimd) is
//! architecturally guaranteed on aarch64, so aarch64 always gets at least
//! the NEON tier; other architectures get the scalar tail. 8/16-bit gather
//! has no SVE hardware instruction (Part 0): narrow keys stay scalar and
//! schema should prefer u32 keys (already the convention).
//!
//! All SVE intrinsics sit behind `stdarch_aarch64_sve` (unstable, enabled
//! crate-wide under `cfg(aarch64)`) and are only reachable after a runtime
//! `is_aarch64_feature_detected!("sve")` probe.

use crate::CpuKernel;

/// `out[i] = src[keys[i]]` over u32 lanes with u32 keys, best available tier.
pub fn gather_u32(src: &[u32], keys: &[u32], out: &mut [u32]) {
    static KERNEL: CpuKernel<unsafe fn(&[u32], &[u32], &mut [u32])> = CpuKernel::new(|| {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("sve") {
                return gather_u32_sve;
            }
            return gather_u32_neon;
        }
        // Portable tail (x86 etc.): the unsafe fn pointer accepts a safe fn.
        #[allow(unreachable_code)]
        gather_u32_scalar
    });
    // SAFETY: the selector probed the required feature for the chosen tier;
    // arguments are valid slices and every tier computes the same mapping.
    unsafe { (KERNEL.get())(src, keys, out) }
}

/// `out[keys[i]] = vals[i]` over u32 lanes with u32 keys, best available tier.
pub fn scatter_u32(keys: &[u32], vals: &[u32], out: &mut [u32]) {
    static KERNEL: CpuKernel<unsafe fn(&[u32], &[u32], &mut [u32])> = CpuKernel::new(|| {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("sve") {
                return scatter_u32_sve;
            }
            return scatter_u32_neon;
        }
        #[allow(unreachable_code)]
        scatter_u32_scalar
    });
    // SAFETY: selector probed features; every tier writes `out[keys[i]] = vals[i]`.
    unsafe { (KERNEL.get())(keys, vals, out) }
}

/// `out[i]` = 1 where `a[i] != b[i]`, else 0 (u32 lanes). The `diff` verb
/// (Part 3 #5) scans these lanes for changed indices; the compare itself is
/// the SIMD-heavy part.
pub fn neq_lanes_u32(a: &[u32], b: &[u32], out: &mut [u32]) {
    static KERNEL: CpuKernel<unsafe fn(&[u32], &[u32], &mut [u32])> = CpuKernel::new(|| {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("sve") {
                return neq_lanes_u32_sve;
            }
            return neq_lanes_u32_neon;
        }
        #[allow(unreachable_code)]
        neq_lanes_u32_scalar
    });
    // SAFETY: selector probed features; all tiers write 0/1 lanes.
    unsafe { (KERNEL.get())(a, b, out) }
}

/// `out[i] = a[i] + c` over u32 lanes, best tier (Part 13.5 dense lift).
pub fn add_const_u32(a: &[u32], c: u32, out: &mut [u32]) {
    static KERNEL: CpuKernel<unsafe fn(&[u32], u32, &mut [u32])> = CpuKernel::new(|| {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("sve") {
                return add_const_u32_sve;
            }
            return add_const_u32_neon;
        }
        #[allow(unreachable_code)]
        add_const_u32_scalar
    });
    // SAFETY: selector probed features; all tiers compute wrapping a+c.
    unsafe { (KERNEL.get())(a, c, out) }
}

// =============================================================================
// SVE tiers (stdarch; verified intrinsic names)
// =============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve")]
unsafe fn gather_u32_sve(src: &[u32], keys: &[u32], out: &mut [u32]) {
    use std::arch::aarch64::{
        svcntw, svld1_gather_u32index_u32, svld1_u32, svst1_u32, svwhilelt_b32_u32,
    };
    unsafe {
        debug_assert_eq!(keys.len(), out.len());
        let n = keys.len();
        let (mut i, vl) = (0usize, svcntw() as usize);
        let sp = src.as_ptr();
        while i < n {
            let pg = svwhilelt_b32_u32(i as u32, n as u32);
            let ks = svld1_u32(pg, keys.as_ptr().add(i));
            let vs = svld1_gather_u32index_u32(pg, sp, ks);
            svst1_u32(pg, out.as_mut_ptr().add(i), vs);
            i += vl;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve")]
unsafe fn scatter_u32_sve(keys: &[u32], vals: &[u32], out: &mut [u32]) {
    use std::arch::aarch64::{
        svcntw, svld1_u32, svst1_scatter_u32index_u32, svwhilelt_b32_u32,
    };
    unsafe {
        debug_assert_eq!(keys.len(), vals.len());
        let n = keys.len();
        let (mut i, vl) = (0usize, svcntw() as usize);
        while i < n {
            let pg = svwhilelt_b32_u32(i as u32, n as u32);
            let ks = svld1_u32(pg, keys.as_ptr().add(i));
            let vs = svld1_u32(pg, vals.as_ptr().add(i));
            svst1_scatter_u32index_u32(pg, out.as_mut_ptr(), ks, vs);
            i += vl;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve")]
unsafe fn neq_lanes_u32_sve(a: &[u32], b: &[u32], out: &mut [u32]) {
    use std::arch::aarch64::{
        svcmpeq_u32, svcntw, svdup_n_u32, svld1_u32, svsel_u32, svst1_u32, svwhilelt_b32_u32,
    };
    unsafe {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len();
        let one = svdup_n_u32(1);
        let zero = svdup_n_u32(0);
        let (mut i, vl) = (0usize, svcntw() as usize);
        while i < n {
            let pg = svwhilelt_b32_u32(i as u32, n as u32);
            let av = svld1_u32(pg, a.as_ptr().add(i));
            let bv = svld1_u32(pg, b.as_ptr().add(i));
            let eq = svcmpeq_u32(pg, av, bv);
            // 0 where equal, 1 where not-equal.
            let lanes = svsel_u32(eq, zero, one);
            svst1_u32(pg, out.as_mut_ptr().add(i), lanes);
            i += vl;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve")]
unsafe fn add_const_u32_sve(a: &[u32], c: u32, out: &mut [u32]) {
    use std::arch::aarch64::{
        svadd_n_u32_x, svcntw, svld1_u32, svst1_u32, svwhilelt_b32_u32,
    };
    unsafe {
        debug_assert_eq!(a.len(), out.len());
        let n = a.len();
        let (mut i, vl) = (0usize, svcntw() as usize);
        while i < n {
            let pg = svwhilelt_b32_u32(i as u32, n as u32);
            let v = svld1_u32(pg, a.as_ptr().add(i));
            let r = svadd_n_u32_x(pg, v, c);
            svst1_u32(pg, out.as_mut_ptr().add(i), r);
            i += vl;
        }
    }
}

// =============================================================================
// NEON tiers (aarch64 baseline)
// =============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn gather_u32_neon(src: &[u32], keys: &[u32], out: &mut [u32]) {
    gather_u32_scalar(src, keys, out)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scatter_u32_neon(keys: &[u32], vals: &[u32], out: &mut [u32]) {
    scatter_u32_scalar(keys, vals, out)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neq_lanes_u32_neon(a: &[u32], b: &[u32], out: &mut [u32]) {
    neq_lanes_u32_scalar(a, b, out)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn add_const_u32_neon(a: &[u32], c: u32, out: &mut [u32]) {
    add_const_u32_scalar(a, c, out)
}

// =============================================================================
// Portable scalar tiers (all architectures)
// =============================================================================

fn gather_u32_scalar(src: &[u32], keys: &[u32], out: &mut [u32]) {
    for (o, &k) in out.iter_mut().zip(keys.iter()) {
        *o = *src.get(k as usize).unwrap_or(&0);
    }
}

fn scatter_u32_scalar(keys: &[u32], vals: &[u32], out: &mut [u32]) {
    for (&k, &v) in keys.iter().zip(vals.iter()) {
        if let Some(slot) = out.get_mut(k as usize) {
            *slot = v;
        }
    }
}

fn neq_lanes_u32_scalar(a: &[u32], b: &[u32], out: &mut [u32]) {
    assert_eq!(a.len(), b.len());
    for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = if x != y { 1 } else { 0 };
    }
}

fn add_const_u32_scalar(a: &[u32], c: u32, out: &mut [u32]) {
    for (o, &v) in out.iter_mut().zip(a.iter()) {
        *o = v.wrapping_add(c);
    }
}

// =============================================================================
// Bitwise ops + compaction filter (Part 3 #4 filter/bit-ops tiers)
// =============================================================================

/// Element-wise bitwise op: `out[i] = a[i] op b[i]` (Not is unary).
pub fn bitwise_u32(op: BitwiseOp, a: &[u32], b: &[u32], out: &mut [u32]) {
    match op {
        BitwiseOp::And => bitwise_and_u32(a, b, out),
        BitwiseOp::Or => bitwise_or_u32(a, b, out),
        BitwiseOp::Xor => bitwise_xor_u32(a, b, out),
        BitwiseOp::Not => bitwise_not_u32(a, out),
    }
}

/// Bitwise operation selector for [`bitwise_u32`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitwiseOp {
    /// Bitwise AND.
    And,
    /// Bitwise OR.
    Or,
    /// Bitwise XOR.
    Xor,
    /// Bitwise NOT (unary; `b` is ignored).
    Not,
}

fn bitwise_and_u32(a: &[u32], b: &[u32], out: &mut [u32]) {
    static KERNEL: CpuKernel<unsafe fn(&[u32], &[u32], &mut [u32])> = CpuKernel::new(|| {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("sve") {
                return and_u32_sve;
            }
            return and_u32_neon;
        }
        #[allow(unreachable_code)]
        and_u32_scalar
    });
    unsafe { (KERNEL.get())(a, b, out) }
}

fn bitwise_or_u32(a: &[u32], b: &[u32], out: &mut [u32]) {
    static KERNEL: CpuKernel<unsafe fn(&[u32], &[u32], &mut [u32])> = CpuKernel::new(|| {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("sve") {
                return or_u32_sve;
            }
            return or_u32_neon;
        }
        #[allow(unreachable_code)]
        or_u32_scalar
    });
    unsafe { (KERNEL.get())(a, b, out) }
}

fn bitwise_xor_u32(a: &[u32], b: &[u32], out: &mut [u32]) {
    static KERNEL: CpuKernel<unsafe fn(&[u32], &[u32], &mut [u32])> = CpuKernel::new(|| {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("sve") {
                return xor_u32_sve;
            }
            return xor_u32_neon;
        }
        #[allow(unreachable_code)]
        xor_u32_scalar
    });
    unsafe { (KERNEL.get())(a, b, out) }
}

fn bitwise_not_u32(a: &[u32], out: &mut [u32]) {
    static KERNEL: CpuKernel<unsafe fn(&[u32], &mut [u32])> = CpuKernel::new(|| {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("sve") {
                return not_u32_sve;
            }
            return not_u32_neon;
        }
        #[allow(unreachable_code)]
        not_u32_scalar
    });
    unsafe { (KERNEL.get())(a, out) }
}

/// Compaction filter: append `src[i]` to `out` where `keep[i] != 0`.
/// Returns the number of kept rows. `out` may be shorter than `src`; it is
/// reused across calls (cleared then extended).
pub fn filter_compact_u32(src: &[u32], keep: &[u32], out: &mut Vec<u32>) -> usize {
    debug_assert_eq!(src.len(), keep.len(), "src/keep length mismatch");
    out.clear();
    let count = compact_count_u32(keep);
    out.reserve(count);
    // Compact writing via the tier (raw-pointer write into reserved capacity).
    let written = compact_into(src, keep, out);
    // SAFETY: written <= capacity (reserved with the count pre-pass) and the
    // kernel initialized exactly the first `written` rows.
    unsafe { out.set_len(written) };
    written
}

/// Count nonzero lanes (for capacity planning) without compacting.
pub fn compact_count_u32(keep: &[u32]) -> usize {
    static KERNEL: CpuKernel<unsafe fn(&[u32]) -> usize> = CpuKernel::new(|| {
        #[cfg(target_arch = "aarch64")]
        {
            // COMPACT is an SVE2 instruction; probing only SVE would SIGILL
            // on SVE-without-SVE2 cores (e.g. A64FX). Fall back below.
            if std::arch::is_aarch64_feature_detected!("sve2") {
                return compact_count_u32_sve;
            }
            return compact_count_u32_neon;
        }
        compact_count_u32_scalar
    });
    // SAFETY: selector probed features; count tiers are pure.
    unsafe { (KERNEL.get())(keep) }
}

/// Write the kept lanes into the front of `out`; returns rows written.
fn compact_into(src: &[u32], keep: &[u32], out: &mut Vec<u32>) -> usize {
    static KERNEL: CpuKernel<unsafe fn(&[u32], &[u32], *mut u32, usize) -> usize> =
        CpuKernel::new(|| {
            #[cfg(target_arch = "aarch64")]
            {
                if std::arch::is_aarch64_feature_detected!("sve2") {
                    return compact_into_u32_sve;
                }
                return compact_into_u32_neon;
            }
            compact_into_u32_scalar
        });
    // SAFETY: `out` is assumed to have capacity >= src.len() (caller reserved
    // with the count pre-pass); the kernel writes at most `capacity` rows.
    let cap = out.capacity();
    let ptr = out.as_mut_ptr();
    unsafe { (KERNEL.get())(src, keep, ptr, cap) }
}

// ---- SVE tiers ---------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve")]
unsafe fn and_u32_sve(a: &[u32], b: &[u32], out: &mut [u32]) {
    use std::arch::aarch64::{svand_u32_x, svcntw, svld1_u32, svst1_u32, svwhilelt_b32_u32};
    unsafe {
        let (mut i, vl) = (0usize, svcntw() as usize);
        while i < a.len() {
            let pg = svwhilelt_b32_u32(i as u32, a.len() as u32);
            let x = svand_u32_x(pg, svld1_u32(pg, a.as_ptr().add(i)), svld1_u32(pg, b.as_ptr().add(i)));
            svst1_u32(pg, out.as_mut_ptr().add(i), x);
            i += vl;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve")]
unsafe fn or_u32_sve(a: &[u32], b: &[u32], out: &mut [u32]) {
    use std::arch::aarch64::{svcntw, svld1_u32, svorr_u32_x, svst1_u32, svwhilelt_b32_u32};
    unsafe {
        let (mut i, vl) = (0usize, svcntw() as usize);
        while i < a.len() {
            let pg = svwhilelt_b32_u32(i as u32, a.len() as u32);
            let x = svorr_u32_x(pg, svld1_u32(pg, a.as_ptr().add(i)), svld1_u32(pg, b.as_ptr().add(i)));
            svst1_u32(pg, out.as_mut_ptr().add(i), x);
            i += vl;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve")]
unsafe fn xor_u32_sve(a: &[u32], b: &[u32], out: &mut [u32]) {
    use std::arch::aarch64::{svcntw, svld1_u32, svst1_u32, sveor_u32_x, svwhilelt_b32_u32};
    unsafe {
        let (mut i, vl) = (0usize, svcntw() as usize);
        while i < a.len() {
            let pg = svwhilelt_b32_u32(i as u32, a.len() as u32);
            let x = sveor_u32_x(pg, svld1_u32(pg, a.as_ptr().add(i)), svld1_u32(pg, b.as_ptr().add(i)));
            svst1_u32(pg, out.as_mut_ptr().add(i), x);
            i += vl;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve")]
unsafe fn not_u32_sve(a: &[u32], out: &mut [u32]) {
    use std::arch::aarch64::{svcntw, sveor_u32_x, svld1_u32, svst1_u32, svdup_n_u32, svwhilelt_b32_u32};
    unsafe {
        let (mut i, vl) = (0usize, svcntw() as usize);
        let ones = svdup_n_u32(u32::MAX);
        while i < a.len() {
            let pg = svwhilelt_b32_u32(i as u32, a.len() as u32);
            let x = sveor_u32_x(pg, svld1_u32(pg, a.as_ptr().add(i)), ones);
            svst1_u32(pg, out.as_mut_ptr().add(i), x);
            i += vl;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve2")]
unsafe fn compact_count_u32_sve(keep: &[u32]) -> usize {
    use std::arch::aarch64::{
        svcmpeq_u32, svcntp_b32, svcntw, svdup_n_u32, svld1_u32, svnot_b_z, svwhilelt_b32_u32,
    };
    unsafe {
        let (mut i, mut count, vl) = (0usize, 0usize, svcntw() as usize);
        let zero = svdup_n_u32(0);
        while i < keep.len() {
            let pg = svwhilelt_b32_u32(i as u32, keep.len() as u32);
            let nonzero = svnot_b_z(pg, svcmpeq_u32(pg, svld1_u32(pg, keep.as_ptr().add(i)), zero));
            count += svcntp_b32(pg, nonzero) as usize;
            i += vl;
        }
        count
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve2")]
unsafe fn compact_into_u32_sve(src: &[u32], keep: &[u32], out: *mut u32, cap: usize) -> usize {
    use std::arch::aarch64::{
        svcmpeq_u32, svcntp_b32, svcntw, svcompact_s32, svdup_n_u32, svld1_u32, svnot_b_z,
        svptrue_b32, svst1_u32, svwhilelt_b32_u32,
    };
    unsafe {
        let _ = cap;
        let (mut i, mut w, vl) = (0usize, 0usize, svcntw() as usize);
        // The predicate's true-count comes from the count tier; we compact and
        // append into the output pointer, tracking `w` via per-iter counts.
        while i < src.len() {
            let pg = svwhilelt_b32_u32(i as u32, src.len() as u32);
            let v = svld1_u32(pg, src.as_ptr().add(i));
            let keepv = svld1_u32(pg, keep.as_ptr().add(i));
            let nonzero = svnot_b_z(pg, svcmpeq_u32(pg, keepv, svdup_n_u32(0)));
            let n_kept = svcntp_b32(pg, nonzero) as usize;
            if n_kept > 0 {
                let compacted = svcompact_s32(nonzero, std::mem::transmute(v));
                svst1_u32(svptrue_b32(), out.add(w), std::mem::transmute(compacted));
                w += n_kept;
            }
            i += vl;
        }
        w
    }
}


// ---- NEON tiers ---------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn and_u32_neon(a: &[u32], b: &[u32], out: &mut [u32]) {
    and_u32_scalar(a, b, out)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn or_u32_neon(a: &[u32], b: &[u32], out: &mut [u32]) {
    or_u32_scalar(a, b, out)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn xor_u32_neon(a: &[u32], b: &[u32], out: &mut [u32]) {
    xor_u32_scalar(a, b, out)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn not_u32_neon(a: &[u32], out: &mut [u32]) {
    not_u32_scalar(a, out)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn compact_count_u32_neon(keep: &[u32]) -> usize {
    compact_count_u32_scalar(keep)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn compact_into_u32_neon(src: &[u32], keep: &[u32], out: *mut u32, cap: usize) -> usize {
    compact_into_u32_scalar(src, keep, out, cap)
}

// ---- Scalar tiers ---------------------------------------------------------

fn and_u32_scalar(a: &[u32], b: &[u32], out: &mut [u32]) {
    assert_eq!(a.len(), b.len());
    for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = x & y;
    }
}

fn or_u32_scalar(a: &[u32], b: &[u32], out: &mut [u32]) {
    assert_eq!(a.len(), b.len());
    for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = x | y;
    }
}

fn xor_u32_scalar(a: &[u32], b: &[u32], out: &mut [u32]) {
    assert_eq!(a.len(), b.len());
    for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = x ^ y;
    }
}

fn not_u32_scalar(a: &[u32], out: &mut [u32]) {
    for (o, &v) in out.iter_mut().zip(a.iter()) {
        *o = !v;
    }
}

fn compact_count_u32_scalar(keep: &[u32]) -> usize {
    keep.iter().filter(|&&k| k != 0).count()
}

fn compact_into_u32_scalar(src: &[u32], keep: &[u32], out: *mut u32, cap: usize) -> usize {
    assert_eq!(src.len(), keep.len());
    let mut w = 0usize;
    for (i, &k) in keep.iter().enumerate() {
        if k != 0 {
            if w >= cap {
                break; // safety net; caller pre-reserved
            }
            unsafe { *out.add(w) = src[i] };
            w += 1;
        }
    }
    w
}
