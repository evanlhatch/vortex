// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod operations;
mod validity;

/// Placeholder encoding marker for [`super::MapArray`].
///
/// The corresponding vtable will be implemented with the canonical map array layout.
#[derive(Clone, Debug, Default)]
pub struct Map;
