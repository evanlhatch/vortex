// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// Placeholder for Vortex's canonical map array.
///
/// Its physical storage layout will be defined when map array support is implemented.
#[derive(Clone, Debug, Default)]
pub struct MapArray;

/// Placeholder for map-array metadata that is independent of child slots.
///
/// Fields will be added with the selected canonical map layout.
#[derive(Clone, Debug, Default)]
pub struct MapData;

/// Placeholder for the inputs used to construct a [`MapArray`].
///
/// Its fields will be defined with the selected canonical map layout.
#[derive(Clone, Debug, Default)]
pub struct MapDataParts;

/// Marker trait for map-array accessors.
///
/// Logical accessors will be added once the physical layout is selected.
pub trait MapArrayExt {}

impl MapArrayExt for MapArray {}
