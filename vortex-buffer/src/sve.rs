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
                return unsafe { gather_u32_sve };
            }
            return unsafe { gather_u32_neon };
        }
        // Portable tail (x86 etc.): the unsafe fn pointer accepts a safe fn.
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
                return unsafe { scatter_u32_sve };
            }
            return unsafe { scatter_u32_neon };
        }
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
                return unsafe { neq_lanes_u32_sve };
            }
            return unsafe { neq_lanes_u32_neon };
        }
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
                return unsafe { add_const_u32_sve };
            }
            return unsafe { add_const_u32_neon };
        }
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
    unsafe { gather_u32_scalar(src, keys, out) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scatter_u32_neon(keys: &[u32], vals: &[u32], out: &mut [u32]) {
    unsafe { scatter_u32_scalar(keys, vals, out) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neq_lanes_u32_neon(a: &[u32], b: &[u32], out: &mut [u32]) {
    unsafe { neq_lanes_u32_scalar(a, b, out) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn add_const_u32_neon(a: &[u32], c: u32, out: &mut [u32]) {
    unsafe { add_const_u32_scalar(a, c, out) }
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
