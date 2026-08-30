// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Varlen-access kernels (flatland REBUILD Part 3 #4, retiered).
//!
//! This module covers ONLY the operations `std::simd` cannot express:
//! indexed gather, indexed scatter, and varlen compaction (SVE2 `COMPACT`).
//! Predicated loops scale to the hardware vector length via
//! `svwhilelt`/`svcntw` — no fixed-width clones.
//!
//! Policy (Part 8, portable-default): **elementwise ops live in
//! [`super::portable`]** (Rust `std::simd`), which lowers to the best
//! fixed-width ISA on every target (NEON on aarch64, AVX2/AVX-512 on x86)
//! and is the default tier everywhere. The SVE tiers below are aarch64
//! accelerators for varlen access, gated by runtime
//! `is_aarch64_feature_detected!("sve"/"sve2")` with NEON (architecturally
//! guaranteed) and scalar fallbacks. 8/16-bit gather has no SVE hardware
//! instruction (Part 0): narrow keys stay scalar and schema should prefer
//! u32 keys (already the convention).
//!
//! All SVE intrinsics sit behind `stdarch_aarch64_sve` (unstable, enabled
//! crate-wide under `cfg(aarch64)`) and are only reachable after a runtime
//! probe.
//!
// SVE register addressing casts block-local indices to the element width;
// flatland blocks are <= 1024 values, so truncation is impossible.
#![allow(
    clippy::cast_possible_truncation,
    reason = "flatland blocks are <= 1024 values; block-local indices cannot exceed u32"
)]
#![allow(
    clippy::missing_transmute_annotations,
    reason = "svcompact_s32 operates on signed lanes; the u32 payload is reinterpreted with identical lane layout"
)]

use crate::CpuKernel;

/// Kernel shape: two u32 slices in, one out (gather/scatter).
type U32Kernel2 = unsafe fn(&[u32], &[u32], &mut [u32]);
/// Count kernel shape: keep lanes in, kept-row count out.
type CountKernel = unsafe fn(&[u32]) -> usize;
/// Compact kernel shape: source, keep mask, output pointer, capacity.
type CompactKernel = unsafe fn(&[u32], &[u32], *mut u32, usize) -> usize;

/// `out[i] = src[keys[i]]` over u32 lanes with u32 keys, best available tier.
pub fn gather_u32(src: &[u32], keys: &[u32], out: &mut [u32]) {
    static KERNEL: CpuKernel<U32Kernel2> = CpuKernel::new(|| {
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
    static KERNEL: CpuKernel<U32Kernel2> = CpuKernel::new(|| {
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

/// `out[i]` = 1 where `a[i] != b[i]`, else 0 (u32 lanes). Delegates to the
/// portable `std::simd` tier (Part 8 portable-default policy) — the `diff`
/// verb (Part 3 #5) scans these lanes for changed indices.
pub fn neq_lanes_u32(a: &[u32], b: &[u32], out: &mut [u32]) {
    super::portable::neq_u32_portable(a, b, out)
}

/// `out[i] = a[i] + c` over u32 lanes (Part 13.5 dense lift), portable tier.
pub fn add_const_u32(a: &[u32], c: u32, out: &mut [u32]) {
    super::portable::add_const_portable(a, c, out)
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
        let len = keys.len();
        let (mut idx, step) = (0usize, svcntw() as usize);
        let src_ptr = src.as_ptr();
        while idx < len {
            let pg = svwhilelt_b32_u32(idx as u32, len as u32);
            let key_lanes = svld1_u32(pg, keys.as_ptr().add(idx));
            let val_lanes = svld1_gather_u32index_u32(pg, src_ptr, key_lanes);
            svst1_u32(pg, out.as_mut_ptr().add(idx), val_lanes);
            idx += step;
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
        let len = keys.len();
        let (mut idx, step) = (0usize, svcntw() as usize);
        while idx < len {
            let pg = svwhilelt_b32_u32(idx as u32, len as u32);
            let key_lanes = svld1_u32(pg, keys.as_ptr().add(idx));
            let val_lanes = svld1_u32(pg, vals.as_ptr().add(idx));
            svst1_scatter_u32index_u32(pg, out.as_mut_ptr(), key_lanes, val_lanes);
            idx += step;
        }
    }
}

// =============================================================================
// NEON tiers (aarch64 baseline; std::simd has no gather/scatter/compact)
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

// =============================================================================
// Portable scalar tiers (all architectures; varlen access only)
// =============================================================================

fn gather_u32_scalar(src: &[u32], keys: &[u32], out: &mut [u32]) {
    for (slot, &key) in out.iter_mut().zip(keys.iter()) {
        *slot = *src.get(key as usize).unwrap_or(&0);
    }
}

fn scatter_u32_scalar(keys: &[u32], vals: &[u32], out: &mut [u32]) {
    for (&key, &val) in keys.iter().zip(vals.iter()) {
        if let Some(slot) = out.get_mut(key as usize) {
            *slot = val;
        }
    }
}

// =============================================================================
// Bitwise selectors + compaction filter (Part 3 #4; std::simd has no varlen
// compact)
// =============================================================================

/// Element-wise bitwise op: `out[i] = a[i] op b[i]` (Not is unary).
/// Portable `std::simd` tier everywhere (Part 8 portable-default policy).
pub fn bitwise_u32(op: BitwiseOp, a: &[u32], b: &[u32], out: &mut [u32]) {
    match op {
        BitwiseOp::And => super::portable::and_u32_portable(a, b, out),
        BitwiseOp::Or => super::portable::or_u32_portable(a, b, out),
        BitwiseOp::Xor => super::portable::xor_u32_portable(a, b, out),
        BitwiseOp::Not => super::portable::not_u32_portable(a, out),
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
    static KERNEL: CpuKernel<CountKernel> = CpuKernel::new(|| {
        #[cfg(target_arch = "aarch64")]
        {
            // COMPACT is an SVE2 instruction; probing only SVE would SIGILL
            // on SVE-without-SVE2 cores (e.g. A64FX). Fall back below.
            if std::arch::is_aarch64_feature_detected!("sve2") {
                return compact_count_u32_sve;
            }
            return compact_count_u32_neon;
        }
        #[allow(unreachable_code)]
        compact_count_u32_scalar
    });
    // SAFETY: selector probed features; count tiers are pure.
    unsafe { (KERNEL.get())(keep) }
}

/// Write the kept lanes into the front of `out`; returns rows written.
fn compact_into(src: &[u32], keep: &[u32], out: &mut Vec<u32>) -> usize {
    static KERNEL: CpuKernel<CompactKernel> = CpuKernel::new(|| {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("sve2") {
                return compact_into_u32_sve;
            }
            return compact_into_u32_neon;
        }
        #[allow(unreachable_code)]
        compact_into_u32_scalar
    });
    // SAFETY: `out` is assumed to have capacity >= src.len() (caller reserved
    // with the count pre-pass); the kernel writes at most `capacity` rows.
    let cap = out.capacity();
    let ptr = out.as_mut_ptr();
    unsafe { (KERNEL.get())(src, keep, ptr, cap) }
}

// ---- SVE2 tiers ---------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve2")]
unsafe fn compact_count_u32_sve(keep: &[u32]) -> usize {
    use std::arch::aarch64::{
        svcmpeq_u32, svcntp_b32, svcntw, svdup_n_u32, svld1_u32, svnot_b_z, svwhilelt_b32_u32,
    };
    unsafe {
        let (mut idx, mut count, step) = (0usize, 0usize, svcntw() as usize);
        let zero = svdup_n_u32(0);
        while idx < keep.len() {
            let pg = svwhilelt_b32_u32(idx as u32, keep.len() as u32);
            let nonzero =
                svnot_b_z(pg, svcmpeq_u32(pg, svld1_u32(pg, keep.as_ptr().add(idx)), zero));
            count += svcntp_b32(pg, nonzero) as usize;
            idx += step;
        }
        count
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "sve2")]
unsafe fn compact_into_u32_sve(src: &[u32], keep: &[u32], out: *mut u32, cap: usize) -> usize {
    use std::arch::aarch64::{
        svcmpeq_u32, svcntp_b32, svcntw, svcompact_s32, svdup_n_u32, svint32_t, svld1_u32,
        svnot_b_z, svptrue_b32, svst1_u32, svuint32_t, svwhilelt_b32_u32,
    };
    unsafe {
        let _ = cap;
        let (mut idx, mut written, step) = (0usize, 0usize, svcntw() as usize);
        while idx < src.len() {
            let pg = svwhilelt_b32_u32(idx as u32, src.len() as u32);
            let value_lanes = svld1_u32(pg, src.as_ptr().add(idx));
            let keep_lanes = svld1_u32(pg, keep.as_ptr().add(idx));
            let nonzero = svnot_b_z(pg, svcmpeq_u32(pg, keep_lanes, svdup_n_u32(0)));
            let kept_here = svcntp_b32(pg, nonzero) as usize;
            if kept_here > 0 {
                // svcompact_s32 operates on signed lanes; the u32 payload is
                // reinterpreted with identical lane layout.
                let compacted: svint32_t = svcompact_s32(
                    nonzero,
                    std::mem::transmute::<svuint32_t, svint32_t>(value_lanes),
                );
                svst1_u32(
                    svptrue_b32(),
                    out.add(written),
                    std::mem::transmute::<svint32_t, svuint32_t>(compacted),
                );
                written += kept_here;
            }
            idx += step;
        }
        written
    }
}

// ---- NEON tiers ---------------------------------------------------------

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

// ---- Scalar tiers (compact only) -----------------------------------------

fn compact_count_u32_scalar(keep: &[u32]) -> usize {
    keep.iter().filter(|&&k| k != 0).count()
}

fn compact_into_u32_scalar(src: &[u32], keep: &[u32], out: *mut u32, cap: usize) -> usize {
    assert_eq!(src.len(), keep.len());
    let mut written = 0usize;
    for (idx, &keep_flag) in keep.iter().enumerate() {
        if keep_flag != 0 {
            if written >= cap {
                break; // safety net; caller pre-reserved
            }
            unsafe { *out.add(written) = src[idx] };
            written += 1;
        }
    }
    written
}
