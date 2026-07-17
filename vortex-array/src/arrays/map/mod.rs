// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Canonical map array scaffolding.
//!
//! The physical layout for maps has not been selected yet. This module only reserves the
//! standard canonical-array structure and public type names.

mod array;
pub use array::MapArray;
pub use array::MapArrayExt;
pub use array::MapData;
pub use array::MapDataParts;

pub(crate) mod compute;

mod vtable;
pub use vtable::Map;
