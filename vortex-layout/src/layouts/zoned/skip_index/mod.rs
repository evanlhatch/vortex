//! Skipping-index interface and implementations.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::aggregates::default_zoned_aggregate_fns;
use super::writer::ZonedLayoutOptions;

pub mod bloom;

/// One definition that supplies a persisted aggregate and registers every read-side component
/// needed to consult it.
///
/// The writer helper [`ZonedLayoutOptions::with_skip_index`] is the explicit per-column declaration
/// seam. Readers call [`SkipIndex::register`] on their session before opening the file.
pub trait SkipIndex: Debug + Send + Sync + 'static {
    /// The aggregate state to persist for `input_dtype`, or `None` when unsupported.
    fn aggregate_fn(&self, input_dtype: &DType) -> Option<AggregateFnRef>;

    /// Register the aggregate, optional probe function, and predicate rewrite as one operation.
    fn register(&self, session: &VortexSession);
}

impl ZonedLayoutOptions {
    /// Add `index` to this zoned writer while retaining the default min/max-style aggregates.
    ///
    /// `WriteStrategyBuilder::with_field_zoned_options` can install the configured options for one
    /// field while retaining the default data layout pipeline.
    pub fn with_skip_index<I: SkipIndex + ?Sized>(
        mut self,
        index: &I,
        input_dtype: &DType,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let aggregate_fn = index
            .aggregate_fn(input_dtype)
            .ok_or_else(|| vortex_err!("skip index does not support input dtype {input_dtype}"))?;

        let mut aggregate_fns = self
            .aggregate_fns
            .take()
            .unwrap_or_else(|| default_zoned_aggregate_fns(input_dtype, session))
            .to_vec();
        if !aggregate_fns.iter().any(|stored| stored == &aggregate_fn) {
            aggregate_fns.push(aggregate_fn);
        }
        self.aggregate_fns = Some(Arc::from(aggregate_fns));
        Ok(self)
    }
}
