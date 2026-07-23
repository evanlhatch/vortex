//! Aggregate functions selected by the zoned layout.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::dtype::DType;
use vortex_session::VortexSession;

pub(in crate::layouts::zoned) mod bloom_filter;
mod min_max;

pub(in crate::layouts::zoned) use bloom_filter::BloomFilter;
pub(in crate::layouts::zoned) use bloom_filter::bloom_contains;
pub(in crate::layouts::zoned) use bloom_filter::i64_value;
use min_max::min_max_aggregate_fns;

pub(super) fn default_zoned_aggregate_fns(
    dtype: &DType,
    session: &VortexSession,
) -> Arc<[AggregateFnRef]> {
    let mut aggregate_fns = Vec::from(min_max_aggregate_fns(dtype));
    if Sum
        .return_dtype(&NumericalAggregateOpts::skip_nans(), dtype)
        .is_some()
    {
        aggregate_fns.push(Sum.bind(NumericalAggregateOpts::skip_nans()));
    }
    aggregate_fns.push(NanCount.bind(EmptyOptions));
    aggregate_fns.push(NullCount.bind(EmptyOptions));

    // Stats from geo extension types are discovered from the registry at runtime instead.
    aggregate_fns.extend(session.aggregate_fns().zone_stat_defaults(dtype));

    aggregate_fns.into()
}

#[cfg(test)]
mod tests {
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::fns::sum::Sum;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::extension::datetime::TimeUnit;
    use vortex_array::extension::datetime::Timestamp;

    use super::BloomFilter;
    use super::bloom_filter::BloomOptions;
    use super::default_zoned_aggregate_fns;

    #[test]
    fn default_aggregates_exclude_bloom_filter() {
        let aggregate_fns =
            default_zoned_aggregate_fns(&PType::I64.into(), &vortex_array::array_session());
        let bloom = BloomFilter.bind(BloomOptions::default());

        assert!(
            aggregate_fns
                .iter()
                .all(|aggregate_fn| aggregate_fn != &bloom)
        );
    }

    #[test]
    fn default_aggregates_include_sum_for_numeric_dtype() {
        let aggregate_fns =
            default_zoned_aggregate_fns(&PType::I32.into(), &vortex_array::array_session());

        assert!(aggregate_fns[2].is::<Sum>());
    }

    #[test]
    fn default_aggregates_skip_sum_for_non_summable_dtype() {
        let dtype = DType::Extension(
            Timestamp::new(TimeUnit::Microseconds, Nullability::Nullable).erased(),
        );
        let aggregate_fns = default_zoned_aggregate_fns(&dtype, &vortex_array::array_session());

        assert!(
            aggregate_fns
                .iter()
                .all(|aggregate_fn| !aggregate_fn.is::<Sum>())
        );
    }
}
