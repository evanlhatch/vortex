// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Tests for the flatland SVE kernels (REBUILD Part 3 #4). These run the
// dispatched tier — on an SVE-capable machine (this host: SVE2) the SVE
// path executes; elsewhere the fallback runs. Both must be correct.

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
