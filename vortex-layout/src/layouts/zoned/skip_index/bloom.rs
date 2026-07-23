//! Bloom skipping index for equality predicates.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::VarBinViewArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::is_root;
use vortex_array::expr::not;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_array::stats::rewrite::StatsRewriteCtx;
use vortex_array::stats::rewrite::StatsRewriteRule;
use vortex_array::stats::session::StatsSessionExt;
use vortex_array::stats::stat;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::super::aggregates::BloomFilter;
use super::super::aggregates::bloom_contains;
pub use super::super::aggregates::bloom_filter::BloomOptions;
use super::super::aggregates::i64_value;
use super::SkipIndex;

/// Bloom skipping index for `i64` equality predicates.
#[derive(Clone, Debug, Default)]
pub struct BloomSkipIndex {
    options: BloomOptions,
}

impl BloomSkipIndex {
    /// Create an index with explicit Bloom tuning.
    pub fn new(options: BloomOptions) -> Self {
        Self { options }
    }

    /// The persisted Bloom options.
    pub fn options(&self) -> &BloomOptions {
        &self.options
    }
}

impl SkipIndex for BloomSkipIndex {
    fn aggregate_fn(&self, input_dtype: &DType) -> Option<AggregateFnRef> {
        BloomFilter
            .return_dtype(&self.options, input_dtype)
            .map(|_| BloomFilter.bind(self.options.clone()))
    }

    fn register(&self, session: &VortexSession) {
        session.aggregate_fns().register(BloomFilter);
        session.scalar_fns().register(BloomContains);
        session.stats().register_rewrite(BloomEqRewrite {
            options: self.options.clone(),
        });
    }
}

/// Probe scalar function: test one `i64` literal against each binary Bloom state.
#[derive(Clone, Debug)]
struct BloomContains;

impl ScalarFnVTable for BloomContains {
    type Options = BloomOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.bloom_contains.i64.v1");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        BloomFilter.serialize(options)
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        BloomFilter.deserialize(metadata, session)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("filter"),
            1 => ChildName::from("needle"),
            _ => unreachable!("bloom_contains has exactly two children"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, args: &[DType]) -> VortexResult<DType> {
        vortex_ensure!(
            matches!(args[0], DType::Binary(_)),
            "bloom filter must be Binary"
        );
        vortex_ensure!(
            matches!(args[1], DType::Primitive(PType::I64, _)),
            "bloom needle must be i64"
        );
        Ok(DType::Bool(args[0].nullability() | args[1].nullability()))
    }

    fn execute(
        &self,
        options: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let filters = args.get(0)?.execute::<VarBinViewArray>(ctx)?;
        let needle_array = args.get(1)?;
        let needle = needle_array
            .as_constant()
            .ok_or_else(|| vortex_err!("bloom needle must be constant"))?;
        let Some(needle) = i64_value(&needle)? else {
            return Ok(ConstantArray::new(
                Scalar::null(DType::Bool(Nullability::Nullable)),
                args.row_count(),
            )
            .into_array());
        };

        let validity = filters.varbinview_validity();
        let valid = validity.execute_mask(filters.len(), ctx)?;
        let mut possible = Vec::with_capacity(filters.len());
        for (idx, is_valid) in valid.iter().enumerate() {
            if is_valid {
                let filter = filters.bytes_at(idx);
                vortex_ensure!(
                    filter.len() == options.bytes().get(),
                    "stored bloom byte length does not match options"
                );
                possible.push(bloom_contains(
                    filter.as_slice(),
                    needle,
                    options.hashes().get(),
                ));
            } else {
                possible.push(false);
            }
        }
        Ok(BoolArray::new(BitBuffer::from_iter(possible), validity).into_array())
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        false
    }
}

/// Equality rewrite that turns a Bloom miss into a zone falsifier.
#[derive(Clone, Debug)]
struct BloomEqRewrite {
    options: BloomOptions,
}

impl StatsRewriteRule for BloomEqRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        Binary.id()
    }

    fn falsify(
        &self,
        expr: &Expression,
        ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>> {
        if *expr.as_::<Binary>() != Operator::Eq {
            return Ok(None);
        }

        let (column, literal) = if is_root(expr.child(0)) && expr.child(1).is::<Literal>() {
            (expr.child(0), expr.child(1))
        } else if is_root(expr.child(1)) && expr.child(0).is::<Literal>() {
            (expr.child(1), expr.child(0))
        } else {
            return Ok(None);
        };
        if !matches!(ctx.return_dtype(column)?, DType::Primitive(PType::I64, _))
            || literal.as_::<Literal>().is_null()
        {
            return Ok(None);
        }

        let filter = stat(column.clone(), BloomFilter.bind(self.options.clone()));
        let contains = BloomContains.new_expr(self.options.clone(), [filter, literal.clone()]);
        Ok(Some(not(contains)))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU8;
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::eq;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;

    use super::BloomOptions;
    use super::BloomSkipIndex;
    use super::SkipIndex;
    use crate::layouts::zoned::zone_map::ZoneMap;

    fn small_options() -> BloomOptions {
        BloomOptions::new(
            NonZeroUsize::new(64).expect("64 is non-zero"),
            NonZeroU8::new(3).expect("3 is non-zero"),
        )
    }

    #[test]
    fn missing_stat_stays_inconclusive() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let index = BloomSkipIndex::new(small_options());
        index.register(&session);
        let predicate = eq(root(), lit(42i64));
        let proof = predicate
            .falsify(
                &DType::Primitive(PType::I64, Nullability::NonNullable),
                &session,
            )?
            .expect("equality has a bloom proof");

        let zone_map = ZoneMap::try_new(
            DType::Primitive(PType::I64, Nullability::NonNullable),
            StructArray::try_new(Vec::<&str>::new().into(), vec![], 2, Validity::NonNullable)?,
            Arc::new([]),
            8,
            16,
        )?;
        assert!(zone_map.prune(&proof, &session)?.all_false());
        Ok(())
    }
}
