// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FLATLAND FORK — the fastlanes donation lives in this subtree
//! (the rest of the crate is upstream 0.85 at rest).
//!
//! Surface map (REBUILD Part 3 #3): [`raw_parts`] — RawParts typed views
//! for FoR (reference + encoded child) and BitPacked (packed bytes + width
//! + offset), completing the cursor surface with
//! `vortex_array::flatland::raw_parts`. [`affine`] — encoded-domain affine
//! transform `dst = v*factor + base` (REBUILD Part 0's 3 sound arms).
//! [`unpack`] — runtime-width FastLanes-layout u32 unpack, SVE tier over the
//! lane dimension (REBUILD Part 3 #8).

pub mod affine;
pub mod diff;
pub mod raw_parts;
pub mod unpack;
