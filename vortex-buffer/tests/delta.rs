// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
//
// Tests for DeltaBuffer<T> (REBUILD Part 17.1).

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
