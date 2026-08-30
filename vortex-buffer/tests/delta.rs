// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Tests for DeltaBuffer<T> (REBUILD Part 17.1).


// Integration-test crate: all fns are tests; short names idiomatic in tests.
#![allow(clippy::tests_outside_test_module)]
#![allow(clippy::min_ident_chars, reason = "short names are idiomatic in test bodies")]


#![allow(clippy::cast_possible_truncation, reason = "flatland u32-key convention in tests/benches")]

use vortex_buffer::DeltaBuffer;

#[test]
fn empty_overlay_blends_to_base() {
    let base: Vec<i64> = (0..10).collect();
    let overlay = DeltaBuffer::new(10, 0);
    let mut out = Vec::new();
    overlay.blend(&base, &mut out);
    assert_eq!(out, base);
    assert_eq!(overlay.patch_count(), 0);
}

#[test]
fn set_and_blend_last_write_wins() {
    let base: Vec<i64> = (0..10).collect();
    let mut overlay = DeltaBuffer::new(10, 0);
    overlay.set(2, 200);
    overlay.set(5, 500);
    overlay.set(2, 250); // last-write-wins
    let mut out = Vec::new();
    overlay.blend(&base, &mut out);
    assert_eq!(out, vec![0, 1, 250, 3, 4, 500, 6, 7, 8, 9]);
    assert_eq!(overlay.patch_count(), 2);
}

#[test]
fn overwrite_same_value_is_semantically_empty() {
    let base: Vec<i64> = (0..10).collect();
    let mut overlay = DeltaBuffer::new(10, 0);
    overlay.set(4, 400);
    assert!(!overlay.is_semantically_empty(&base));
    overlay.set(4, 4); // back to base value
    assert!(overlay.is_semantically_empty(&base));
}

#[test]
fn u32_and_f64_types() {
    let base: Vec<u32> = (0..8).collect();
    let mut o = DeltaBuffer::new(8, 0);
    o.set(0, 99);
    let mut out = Vec::new();
    o.blend(&base, &mut out);
    assert_eq!(out[0], 99);
    assert_eq!(out[7], 7);

    let fbase: Vec<f64> = (0..4).map(|i| i as f64 * 0.5).collect();
    let mut fo = DeltaBuffer::new(4, 0.0);
    fo.set(1, 9.5);
    let mut fout = Vec::new();
    fo.blend(&fbase, &mut fout);
    assert_eq!(fout, vec![0.0, 9.5, 1.0, 1.5]);
}

// ── Fuzz ingress (REBUILD Part 12: delta apply boundary) ──────────────────
// Seeded structural fuzzing of the delta-apply ingress — a bolero harness
// needs dependency sign-off, so these run deterministic pseudo-random cases
// instead. Same ingress surface: blend ≡ per-row select, all densities.

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
}

#[test]
fn fuzz_blend_matches_per_row_select() {
    for seed in 0..256u64 {
        let mut state = seed | 1;
        let mut rng = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state >> 33
        };
        let rows = (seed as usize % 300) + 1;
        let base: Vec<u64> = (0..rows as u64).map(|i| i.wrapping_mul(0x9E3779B1)).collect();
        let mut overlay = DeltaBuffer::<u64>::new(rows, 0);
        // Patch density sweeps: sparse (1/64), medium (1/4), dense (1/1).
        for density in [64u64, 4, 1] {
            for row in 0..rows {
                if rng() % density == 0 {
                    overlay.set(row, base[row].wrapping_add(row as u64).wrapping_mul(7));
                }
            }
            // Adversarial: sometimes overwrite the same row (last-write-wins).
            if rows > 0 {
                let hot = (rng() as usize) % rows;
                overlay.set(hot, u64::MAX);
            }

            let mut out = Vec::with_capacity(rows);
            overlay.blend(&base, &mut out);
            let reference: Vec<u64> = (0..rows)
                .map(|i| if overlay.is_patched(i) { overlay.value(i) } else { base[i] })
                .collect();
            assert_eq!(out, reference, "blend ≡ select (seed {seed}, density {density})");
            assert_eq!(out.len(), rows, "out length");
    
        }
    }
}
