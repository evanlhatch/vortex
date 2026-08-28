// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FLATLAND FORK — everything flatland donated lives in this subtree
//! (the rest of the crate is upstream 0.85 at rest).
//!
//! Surface map (REBUILD Part 3):
//! - [`verbs`]: diff / scatter / scatter_in_place / group_indices — the
//!   change-detection and indexed-write verbs, with SVE fast paths.
//! - [`raw_parts`]: `RawParts<T>` typed zero-copy physical views
//!   (Primitive/Constant/Bool/Dict; FoR/BitPacked live in
//!   `vortex_fastlanes::flatland`).
//! - stats-clear guard: [`crate::ArrayRef::try_buffer_mut`] +
//!   `crate::stats::ArrayStats::clear_value_stats` (in-crate, on the type
//!   they mutate).
//! - `Patches::merge_in_place` + `ArrayRef::set_len` (in-crate, on
//!   `crate::patches::Patches`).
//! - `inherit_subset_stats` on slice/filter/take (`crate::ArrayRef`).

pub mod raw_parts;
pub mod verbs;
