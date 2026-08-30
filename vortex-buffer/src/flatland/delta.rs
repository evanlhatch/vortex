// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Dense delta overlay (flatland REBUILD Part 17.1, scope-trimmed).
//!
//! [`DeltaBuffer`] is the fork-native shape that matches the engine's hot
//! SmallVec semantics with a plain buffer: a dense presence
//! [`BitBufferMut`] plus raw value [`BufferMut`]. The hot delta is NEVER an
//! encoded `ArrayRef` (REBUILD Part 17.0 guardrail #1) — this is the buffer
//! form; the encoded array exists only at the consumer/materialization
//! bridge.
//!
//! Write = bit-set + value write (O(1), no alloc under capacity, last-write-
//! wins). Read-merge = branchless `blend(density[i], delta[i], base[i])` with
//! no decode and no per-row branching. Compaction/consumer conversion is a
//! separate consume step.
//!
//! Capacity discipline: `values` is dense up to `len`; `presence` tracks the
//! logical row length. The block is cache-resident by construction (Part 0
//! chunks).

use std::marker::PhantomData;

use crate::BitBufferMut;
use crate::BufferMut;

/// Dense value overlay over an immutable base column.
///
/// `T` is the element type; `rows` is the logical row count. Presence is a
/// 1-bit flag per row; values are stored densely at their index.
pub struct DeltaBuffer<T> {
    presence: BitBufferMut,
    values: BufferMut<T>,
    rows: usize,
    _marker: PhantomData<T>,
}

impl<T: Copy> DeltaBuffer<T> {
    /// Create an overlay for a column of `rows` rows with a shadow value.
    /// Unwritten rows read `shadow` (gated on `is_patched`); callers pass the
    /// base 0 or the base row-0 value.
    pub fn new(rows: usize, shadow: T) -> Self {
        let mut values = BufferMut::with_capacity(rows);
        // Fill to length `rows` so direct indexing is valid.
        let fill = [shadow; 64];
        let mut filled = 0usize;
        while filled < rows {
            let take = (rows - filled).min(64);
            values.extend_from_slice(&fill[..take]);
            filled += take;
        }
        Self {
            presence: BitBufferMut::new_unset(rows),
            values,
            rows,
            _marker: PhantomData,
        }
    }

    /// Logical row count.
    #[inline]
    pub fn len(&self) -> usize {
        self.rows
    }

    /// Whether the overlay covers zero rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// Number of hot (written) rows — word-level popcount over the presence
    /// bitmap, not per-row bit tests (called per overlay per tick).
    #[inline]
    pub fn patch_count(&self) -> usize {
        let bytes = self.presence.as_slice();
        let mut count: usize = bytes.iter().map(|b| b.count_ones() as usize).sum();
        // Zero any padding bits past `rows` in the final byte.
        if self.rows % 8 != 0 && !bytes.is_empty() {
            count -= (bytes[bytes.len() - 1] >> (self.rows % 8)).count_ones() as usize;
        }
        count
    }

    /// Whether row `i` has an overlay value.
    #[inline]
    pub fn is_patched(&self, i: usize) -> bool {
        self.presence.value(i)
    }

    /// Overlay value at row `i` (only valid if `is_patched(i)`).
    #[inline]
    pub fn value(&self, i: usize) -> T {
        // The value buffer is dense to `rows`; unwritten rows hold whatever
        // was there — callers gate reads on `is_patched`.
        self.values[i]
    }

    /// Set `row = value`, last-write-wins. O(1), no alloc under capacity.
    pub fn set(&mut self, row: usize, value: T) {
        debug_assert!(row < self.rows, "row {row} out of bounds for {self:?}");
        self.presence.set_to(row, true);
        self.values[row] = value;
    }

    /// Branchless read-merge: `out[i] = if patched(i) { delta[i] } else { base[i] }`.
    ///
    /// `out` is grown to `self.len()` (reused across calls); `base` must have
    /// at least `self.len()` rows. No decode, no per-row branch: the blend is
    /// a select on the presence bits.
    /// Copy-then-overwrite merge: `out[i] = if patched(i) { delta[i] } else { base[i] }`.
    ///
    /// `out` is grown to `self.len()` (reused across calls); `base` must have
    /// at least `self.len()` rows. One base copy + O(patches) overwrites by
    /// walking the presence words with trailing-zero skips — the Part 17.0
    /// bench gate (blend within noise of a raw Vec slice-merge) drove this
    /// shape: a per-row presence test at low patch density is 2× slower than
    /// the copy baseline.
    pub fn blend(&self, base: &[T], out: &mut Vec<T>) {
        debug_assert!(base.len() >= self.rows);
        out.clear();
        out.extend_from_slice(&base[..self.rows]);
        // Presence words: 8 flags per byte; jump straight to the set bits.
        let presence_bytes = self.presence.as_slice();
        for (byte_idx, &byte) in presence_bytes.iter().enumerate() {
            if byte == 0 {
                continue;
            }
            let mut word = byte;
            while word != 0 {
                let bit = word.trailing_zeros() as usize;
                word &= word - 1; // clear the lowest set bit
                let row = byte_idx * 8 + bit;
                if row < self.rows {
                    out[row] = self.values.as_slice()[row];
                }
            }
        }
    }

    /// Consume into a dense `Vec<T>` (materialization bridge input).
    pub fn into_dense(self, base: &[T], out: &mut Vec<T>) {
        self.blend(base, out);
    }
}

impl<T: Copy + PartialEq> DeltaBuffer<T> {
    /// Whether the overlay is semantically empty (every patched row equals
    /// its base value) — used to drop no-op overlays before promotion.
    pub fn is_semantically_empty(&self, base: &[T]) -> bool {
        (0..self.rows).all(|i| !self.presence.value(i) || self.values[i] == base[i])
    }
}

impl<T: Copy> std::fmt::Debug for DeltaBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeltaBuffer")
            .field("rows", &self.rows)
            .field("patched", &self.patch_count())
            .finish()
    }
}
