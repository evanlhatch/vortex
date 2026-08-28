// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FLATLAND FORK — everything flatland donated lives in this subtree.
//!
//! The rest of this crate is upstream 0.85 at rest. Fork additions are
//! deliberately isolated here so forks/upstream diffs are greppable and a
//! rebase never interleaves our surface with upstream's.
//!
//! Surface map (REBUILD Part 3):
//! - [`sve`]: SVE tiers (gather/scatter/neq-lanes/add-const/bit-ops/
//!   filter-compact), once-per-process CpuKernel dispatch SVE→NEON→scalar.
//! - [`delta`]: [`super::DeltaBuffer`] — dense presence + value overlay with
//!   branchless blend (Part 17.1).

pub mod delta;
pub mod portable;
pub mod sve;
