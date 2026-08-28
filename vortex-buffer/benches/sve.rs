// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// aarch64 benches for the flatland SVE tiers (REBUILD Part 8: "every fork
// donation lands with an aarch64 bench proving parity-or-better"). Run on
// SVE-capable hardware (this host: SVE2):
//
//     cargo bench -p vortex-buffer --bench sve
//
// The SVE tier must be >= parity with the scalar tier (the point of the
// donation). Setup (output buffer alloc) is identical on both sides, so
// the ratio is a valid parity comparison.

use std::hint::black_box;

use divan::Bencher;
use vortex_buffer::sve;
use vortex_buffer::DeltaBuffer;

fn gen_u32(n: usize) -> Vec<u32> {
    (0..n as u32).collect()
}

fn gather_scalar(src: &[u32], keys: &[u32], out: &mut [u32]) {
    for (o, &k) in out.iter_mut().zip(keys.iter()) {
        *o = *src.get(k as usize).unwrap_or(&0);
    }
}

fn scatter_scalar(keys: &[u32], vals: &[u32], out: &mut [u32]) {
    for (&k, &v) in keys.iter().zip(vals.iter()) {
        if let Some(slot) = out.get_mut(k as usize) {
            *slot = v;
        }
    }
}

fn neq_scalar(a: &[u32], b: &[u32], out: &mut [u32]) {
    for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = if x != y { 1 } else { 0 };
    }
}

fn add_scalar(a: &[u32], c: u32, out: &mut [u32]) {
    for (o, &v) in out.iter_mut().zip(a.iter()) {
        *o = v.wrapping_add(c);
    }
}

const N: usize = 1 << 20;

fn inputs() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let src = gen_u32(N);
    let keys: Vec<u32> = (0..N as u32).map(|i| (i * 2654435761) % N as u32).collect();
    let vals = gen_u32(N);
    (src, keys, vals)
}

fn neq_inputs() -> (Vec<u32>, Vec<u32>) {
    let a = gen_u32(N);
    let b: Vec<u32> = a.iter().map(|&v| if v % 3 == 0 { v + 1 } else { v }).collect();
    (a, b)
}

fn main() {
    divan::main();
}

#[divan::bench]
fn gather_sve(b: Bencher) {
    let (src, keys, _) = inputs();
    b.bench(|| {
        let mut out = vec![0u32; N];
        sve::gather_u32(black_box(&src), black_box(&keys), black_box(&mut out));
    });
}

#[divan::bench]
fn gather_scalar_direct(b: Bencher) {
    let (src, keys, _) = inputs();
    b.bench(|| {
        let mut out = vec![0u32; N];
        gather_scalar(black_box(&src), black_box(&keys), black_box(&mut out));
    });
}

#[divan::bench]
fn scatter_sve(b: Bencher) {
    let (_, keys, vals) = inputs();
    b.bench(|| {
        let mut out = vec![0u32; N];
        sve::scatter_u32(black_box(&keys), black_box(&vals), black_box(&mut out));
    });
}

#[divan::bench]
fn scatter_scalar_direct(b: Bencher) {
    let (_, keys, vals) = inputs();
    b.bench(|| {
        let mut out = vec![0u32; N];
        scatter_scalar(black_box(&keys), black_box(&vals), black_box(&mut out));
    });
}

#[divan::bench]
fn neq_lanes_sve(b: Bencher) {
    let (a, bb) = neq_inputs();
    b.bench(|| {
        let mut out = vec![0u32; N];
        sve::neq_lanes_u32(black_box(&a), black_box(&bb), black_box(&mut out));
    });
}

#[divan::bench]
fn neq_lanes_scalar_direct(b: Bencher) {
    let (a, bb) = neq_inputs();
    b.bench(|| {
        let mut out = vec![0u32; N];
        neq_scalar(black_box(&a), black_box(&bb), black_box(&mut out));
    });
}

#[divan::bench]
fn add_const_sve(b: Bencher) {
    let (a, _, _) = inputs();
    b.bench(|| {
        let mut out = vec![0u32; N];
        sve::add_const_u32(black_box(&a), black_box(7), black_box(&mut out));
    });
}

#[divan::bench]
fn add_const_scalar_direct(b: Bencher) {
    let (a, _, _) = inputs();
    b.bench(|| {
        let mut out = vec![0u32; N];
        add_scalar(black_box(&a), black_box(7), black_box(&mut out));
    });
}

// ── Bitwise ops: dispatched vs portable direct vs scalar baseline ─────────

fn bitwise_inputs() -> (Vec<u32>, Vec<u32>) {
    let a = gen_u32(N);
    let b: Vec<u32> = a.iter().map(|&v| v.rotate_left(7) ^ 0x55AA55AA).collect();
    (a, b)
}

#[divan::bench]
fn and_dispatched(b: Bencher) {
    let (a, c) = bitwise_inputs();
    b.bench(|| {
        let mut out = vec![0u32; N];
        sve::bitwise_u32(sve::BitwiseOp::And, black_box(&a), black_box(&c), black_box(&mut out));
    });
}

#[divan::bench]
fn and_portable_direct(b: Bencher) {
    let (a, c) = bitwise_inputs();
    b.bench(|| {
        let mut out = vec![0u32; N];
        vortex_buffer::portable::and_u32(black_box(&a), black_box(&c), black_box(&mut out));
    });
}

#[divan::bench]
fn xor_dispatched(b: Bencher) {
    let (a, c) = bitwise_inputs();
    b.bench(|| {
        let mut out = vec![0u32; N];
        sve::bitwise_u32(sve::BitwiseOp::Xor, black_box(&a), black_box(&c), black_box(&mut out));
    });
}

#[divan::bench]
fn not_dispatched(b: Bencher) {
    let (a, _) = bitwise_inputs();
    b.bench(|| {
        let mut out = vec![0u32; N];
        sve::bitwise_u32(sve::BitwiseOp::Not, black_box(&a), black_box(&a), black_box(&mut out));
    });
}

fn and_scalar_direct(a: &[u32], c: &[u32], out: &mut [u32]) {
    for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(c.iter()) {
        *o = x & y;
    }
}

#[divan::bench]
fn and_scalar_direct_bench(b: Bencher) {
    let (a, c) = bitwise_inputs();
    b.bench(|| {
        let mut out = vec![0u32; N];
        and_scalar_direct(black_box(&a), black_box(&c), black_box(&mut out));
    });
}

// ── Compaction filter (SVE2 COMPACT tier) ──────────────────────────────────

fn keep_inputs() -> (Vec<u32>, Vec<u32>) {
    let src = gen_u32(N);
    // ~50% keep rate, mixed lanes.
    let keep: Vec<u32> = (0..N as u32).map(|i| (i * 2654435761 >> 13) % 2).collect();
    (src, keep)
}

#[divan::bench]
fn filter_compact(b: Bencher) {
    let (src, keep) = keep_inputs();
    b.bench(|| {
        let mut out = Vec::with_capacity(N);
        sve::filter_compact_u32(black_box(&src), black_box(&keep), black_box(&mut out));
    });
}

// ── DeltaBuffer blend vs raw Vec slice-merge baseline (Part 17.0 gate) ────

fn delta_inputs() -> (Vec<u32>, DeltaBuffer<u32>, Vec<usize>) {
    let base = gen_u32(N);
    // Patch 1/64 of rows (realistic ECS overlay density).
    let patch_rows: Vec<usize> = (0..N).step_by(64).collect();
    let mut delta = DeltaBuffer::<u32>::new(N, 0);
    for &r in &patch_rows {
        delta.set(r, r as u32 ^ 0xABCD);
    }
    (base, delta, patch_rows)
}

#[divan::bench]
fn delta_blend(b: Bencher) {
    let (base, delta, _) = delta_inputs();
    b.bench(|| {
        let mut out = Vec::with_capacity(N);
        delta.blend(black_box(&base), black_box(&mut out));
    });
}

/// Part 17.0 gate baseline: a SmallVec-style raw Vec merge where patched
/// rows overwrite after a plain base copy. `blend` must be within noise of
/// this — the overlay's cost is the copy, not the merge structure.
#[divan::bench]
fn blend_raw_vec_baseline(b: Bencher) {
    let (base, delta, _) = delta_inputs();
    b.bench(|| {
        let mut out = base.clone();
        for i in 0..delta.len() {
            if delta.is_patched(i) {
                out[i] = delta.value(i);
            }
        }
        black_box(&out);
    });
}
