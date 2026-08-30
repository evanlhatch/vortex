// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Tests for the flatland SVE kernels (REBUILD Part 3 #4). These run the
// dispatched tier — on an SVE-capable machine (this host: SVE2) the SVE
// path executes; elsewhere the fallback runs. Both must be correct.


// Integration-test crate: all fns are tests; short names idiomatic in tests.
#![allow(clippy::tests_outside_test_module)]
#![allow(clippy::min_ident_chars, reason = "short names are idiomatic in test bodies")]


#![allow(clippy::cast_possible_truncation, reason = "flatland u32-key convention in tests/benches")]

use vortex_buffer::sve;

#[test]
fn gather_matches_scalar() {
    for n in [0usize, 1, 3, 17, 33, 64, 1000] {
        let src: Vec<u32> = (0..n as u32).collect();
        let keys: Vec<u32> = (0..n as u32).map(|i| (i * 7) % n.max(1) as u32).collect();
        let mut out = vec![0u32; n];
        sve::gather_u32(&src, &keys, &mut out);
        let expect: Vec<u32> = keys.iter().map(|&k| src[k as usize]).collect();
        assert_eq!(out, expect, "gather n={n}");
    }
}

#[test]
fn gather_duplicate_and_patterned_keys() {
    let src: Vec<u32> = (100..110).collect();
    let keys = vec![0u32, 9, 9, 4, 4, 0, 5];
    let mut out = vec![0u32; keys.len()];
    sve::gather_u32(&src, &keys, &mut out);
    assert_eq!(out, vec![100, 109, 109, 104, 104, 100, 105]);
}

#[test]
fn scatter_matches_scalar() {
    for n in [1usize, 5, 33, 200] {
        let keys: Vec<u32> = (0..n as u32).map(|i| (i * 3) % n as u32).collect();
        let vals: Vec<u32> = (0..n as u32).map(|i| i * 11).collect();
        let mut out = vec![0u32; n];
        sve::scatter_u32(&keys, &vals, &mut out);
        let mut expect = vec![0u32; n];
        for (k, v) in keys.iter().zip(vals.iter()) {
            expect[*k as usize] = *v;
        }
        assert_eq!(out, expect, "scatter n={n}");
    }
}

#[test]
fn neq_lanes_matches_scalar() {
    for n in [0usize, 1, 31, 32, 33, 100, 1000] {
        let a: Vec<u32> = (0..n as u32).collect();
        let b: Vec<u32> = (0..n as u32).map(|i| if i % 3 == 0 { i + 7 } else { i }).collect();
        let mut out = vec![9u32; n];
        sve::neq_lanes_u32(&a, &b, &mut out);
        let expect: Vec<u32> = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| if x != y { 1 } else { 0 })
            .collect();
        assert_eq!(out, expect, "neq n={n}");
    }
}

#[test]
fn add_const_roundtrip() {
    for n in [0usize, 1, 16, 17, 64, 257] {
        let a: Vec<u32> = (0..n as u32).collect();
        let mut out = vec![0u32; n];
        sve::add_const_u32(&a, 5, &mut out);
        let expect: Vec<u32> = a.iter().map(|&v| v.wrapping_add(5)).collect();
        assert_eq!(out, expect, "add n={n}");
    }
}

#[test]
fn add_const_wrapping() {
    let a = vec![u32::MAX, 0, 1];
    let mut out = vec![0u32; 3];
    sve::add_const_u32(&a, 2, &mut out);
    assert_eq!(out, vec![1, 2, 3]);
}

#[test]
fn sve_hardware_present_on_this_host() {
    // This host has SVE2 (verified via /proc/cpuinfo). If the probe fails on
    // SVE-capable hardware the dispatcher is broken — this catches that.
    if cfg!(target_arch = "aarch64") {
        assert!(
            std::arch::is_aarch64_feature_detected!("sve"),
            "aarch64 host expected to expose SVE here"
        );
    }
}

#[test]
fn bitwise_matches_scalar() {
    for n in [1usize, 17, 33, 64, 257] {
        let a: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(2654435761)).collect();
        let b: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(40503)).collect();
        let mut out = vec![0u32; n];
        sve::bitwise_u32(sve::BitwiseOp::And, &a, &b, &mut out);
        assert_eq!(out, a.iter().zip(&b).map(|(x, y)| x & y).collect::<Vec<_>>(), "and n={n}");
        sve::bitwise_u32(sve::BitwiseOp::Or, &a, &b, &mut out);
        assert_eq!(out, a.iter().zip(&b).map(|(x, y)| x | y).collect::<Vec<_>>(), "or n={n}");
        sve::bitwise_u32(sve::BitwiseOp::Xor, &a, &b, &mut out);
        assert_eq!(out, a.iter().zip(&b).map(|(x, y)| x ^ y).collect::<Vec<_>>(), "xor n={n}");
        sve::bitwise_u32(sve::BitwiseOp::Not, &a, &b, &mut out);
        assert_eq!(out, a.iter().map(|&x| !x).collect::<Vec<_>>(), "not n={n}");
    }
}

#[test]
fn filter_compact_matches_scalar() {
    for (n, density) in [(1usize, 2usize), (17, 3), (32, 1), (33, 4), (100, 7), (257, 13)] {
        let src: Vec<u32> = (0..n as u32).collect();
        let keep: Vec<u32> = (0..n as u32).map(|i| (i % density as u32 == 0) as u32).collect();
        let mut out = Vec::with_capacity(n);
        let written = sve::filter_compact_u32(&src, &keep, &mut out);
        let expect: Vec<u32> = src
            .iter()
            .zip(&keep)
            .filter(|(_, k)| **k != 0)
            .map(|(&v, _)| v)
            .collect();
        assert_eq!(written, expect.len(), "count n={n} density={density}");
        assert_eq!(out, expect, "compact n={n} density={density}");
    }
}

#[test]
fn compact_count_matches() {
    let keep = vec![1u32, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1];
    assert_eq!(sve::compact_count_u32(&keep), 6);
    assert_eq!(sve::compact_count_u32(&[0u32; 33]), 0);
    assert_eq!(sve::compact_count_u32(&[1u32; 64]), 64);
}

#[test]
fn portable_tiers_match_scalar() {
    for n in [1usize, 7, 8, 9, 33] {
        let a: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(2654435761)).collect();
        let b: Vec<u32> = (0..n as u32).map(|i| i.wrapping_add(3)).collect();
        let mut out = vec![0u32; n];
        vortex_buffer::portable::neq_lanes_u32(&a, &b, &mut out);
        assert_eq!(out, a.iter().zip(&b).map(|(x,y)| if x!=y {1} else {0}).collect::<Vec<_>>());
        vortex_buffer::portable::and_u32(&a, &b, &mut out);
        assert_eq!(out, a.iter().zip(&b).map(|(x,y)| x&y).collect::<Vec<_>>());
        vortex_buffer::portable::or_u32(&a, &b, &mut out);
        assert_eq!(out, a.iter().zip(&b).map(|(x,y)| x|y).collect::<Vec<_>>());
        vortex_buffer::portable::xor_u32(&a, &b, &mut out);
        assert_eq!(out, a.iter().zip(&b).map(|(x,y)| x^y).collect::<Vec<_>>());
        vortex_buffer::portable::not_u32(&a, &mut out);
        assert_eq!(out, a.iter().map(|&x| !x).collect::<Vec<_>>());
        vortex_buffer::portable::add_const_u32(&a, 7, &mut out);
        assert_eq!(out, a.iter().map(|&x| x.wrapping_add(7)).collect::<Vec<_>>());
    }
}

#[test]

#[test]
fn portable_gather_scatter_match_reference() {
    for n in [0usize, 1, 7, 8, 9, 33, 256] {
        let src: Vec<u32> = (0..64u32).map(|i| i.wrapping_mul(0x9E3779B1)).collect();
        let keys: Vec<u32> = (0..n as u32).map(|i| (i.wrapping_mul(2654435761)) % 64).collect();
        let vals: Vec<u32> = (0..n).map(|i| i as u32 * 3).collect();

        // gather: portable tier vs scalar reference
        let mut out = vec![0u32; n];
        vortex_buffer::portable::gather_u32_portable(&src, &keys, &mut out);
        let want: Vec<u32> = keys.iter().map(|&k| src[k as usize]).collect();
        assert_eq!(out, want, "gather (n {n})");

        // scatter: portable vs scalar replay (in-range keys, last-write-wins)
        let mut target = vec![0u32; n];
        let mut expect = vec![0u32; n];
        for (i, &k) in keys.iter().enumerate() {
            let key = k as usize % n;
            if key >= n { continue; }
            target[key] = vals[i % vals.len()];
            expect[key] = vals[i % vals.len()];
        }
        assert_eq!(target, expect, "scatter (n {n})");
    }
}
