// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! One import surface for the flatland engine.
//!
//!     use vortex_array::flatland::prelude::*;
//!
//! Everything the engine touches at the array level: the read-side RawParts
//! views, the write-side verb set, and the in-place mutation surface.
//! Replace ~238 scattered `use vortex_*` paths with this line.

pub use crate::flatland::raw_parts::{
    BoolParts, ChunkedParts, ConstantParts, DictParts, PrimitiveParts, RawParts,
};
pub use crate::flatland::verbs::{
    diff, group_indices, scatter, scatter_in_place, GroupIndices,
};
pub use crate::patches::Patches;
