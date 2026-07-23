//! Skipping-index interface and Bloom-filter implementation.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::num::NonZeroU8;
use std::num::NonZeroUsize;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnId;
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

use super::writer::ZonedLayoutOptions;
use super::writer::default_zoned_aggregate_fns;

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

/// Bloom-filter tuning persisted as aggregate metadata.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BloomOptions {
    bytes: NonZeroUsize,
    hashes: NonZeroU8,
}

impl BloomOptions {
    /// Create bloom options with a fixed number of bytes and hash probes per zone.
    pub fn new(bytes: NonZeroUsize, hashes: NonZeroU8) -> Self {
        Self { bytes, hashes }
    }

    /// Bytes stored for each zone.
    pub fn bytes(&self) -> NonZeroUsize {
        self.bytes
    }

    /// Hash probes performed for each inserted or tested value.
    pub fn hashes(&self) -> NonZeroU8 {
        self.hashes
    }
}

impl Default for BloomOptions {
    fn default() -> Self {
        Self {
            // Eight bits per row at the default 8192-row zone size.
            bytes: NonZeroUsize::new(8192).unwrap_or(NonZeroUsize::MIN),
            hashes: NonZeroU8::new(5).unwrap_or(NonZeroU8::MIN),
        }
    }
}

impl Display for BloomOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "bytes={},hashes={}", self.bytes, self.hashes)
    }
}

/// Bloom skipping index for `i64` equality predicates.
#[derive(Clone, Debug, Default)]
pub struct BloomSkipIndex {
    options: BloomOptions,
}

impl BloomSkipIndex {
    /// Create an index with explicit bloom tuning.
    pub fn new(options: BloomOptions) -> Self {
        Self { options }
    }

    /// The persisted bloom options.
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

/// Aggregate that stores one fixed-size bloom bitset as a `Binary` scalar for every zone.
#[derive(Clone, Debug)]
struct BloomFilter;

/// In-memory bloom accumulator. Only the bitset is persisted.
struct BloomPartial {
    bits: Vec<u8>,
    hashes: u8,
}

impl AggregateFnVTable for BloomFilter {
    type Options = BloomOptions;
    type Partial = BloomPartial;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.bloom_filter.i64.v1");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        let bytes = u32::try_from(options.bytes.get())?;
        let mut metadata = bytes.to_le_bytes().to_vec();
        metadata.push(options.hashes.get());
        Ok(Some(metadata))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        vortex_ensure!(metadata.len() == 5, "invalid bloom metadata length");
        let bytes = u32::from_le_bytes([metadata[0], metadata[1], metadata[2], metadata[3]]);
        Ok(BloomOptions::new(
            NonZeroUsize::new(bytes as usize)
                .ok_or_else(|| vortex_err!("bloom byte length must be non-zero"))?,
            NonZeroU8::new(metadata[4])
                .ok_or_else(|| vortex_err!("bloom hash count must be non-zero"))?,
        ))
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        matches!(input_dtype, DType::Primitive(PType::I64, _))
            .then_some(DType::Binary(Nullability::NonNullable))
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        options: &Self::Options,
        _input_dtype: &DType,
    ) -> VortexResult<Self::Partial> {
        Ok(BloomPartial {
            bits: vec![0; options.bytes.get()],
            hashes: options.hashes.get(),
        })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        if other.is_null() {
            return Ok(());
        }
        let other = other
            .as_binary()
            .value()
            .ok_or_else(|| vortex_err!("non-null bloom partial has no bytes"))?;
        vortex_ensure!(
            partial.bits.len() == other.len(),
            "bloom partial length mismatch"
        );
        for (dst, src) in partial.bits.iter_mut().zip(other.as_slice()) {
            *dst |= *src;
        }
        Ok(())
    }

    fn to_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        Ok(Scalar::binary(
            partial.bits.clone(),
            Nullability::NonNullable,
        ))
    }

    fn reset(&self, partial: &mut Self::Partial) {
        partial.bits.fill(0);
    }

    fn is_saturated(&self, partial: &Self::Partial) -> bool {
        partial.bits.iter().all(|byte| *byte == u8::MAX)
    }

    fn accumulate(
        &self,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        match batch {
            Columnar::Constant(constant) => {
                if let Some(value) = i64_value(constant.scalar())? {
                    bloom_insert(&mut partial.bits, value, partial.hashes);
                }
            }
            Columnar::Canonical(canonical) => {
                let primitive = canonical.as_primitive();
                let values = primitive.as_slice::<i64>();
                let validity = primitive.validity()?.execute_mask(values.len(), ctx)?;
                for (&value, valid) in values.iter().zip(validity.iter()) {
                    if valid {
                        bloom_insert(&mut partial.bits, value, partial.hashes);
                    }
                }
            }
        }
        Ok(())
    }

    fn finalize(&self, partials: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(&self, partial: &Self::Partial) -> VortexResult<Scalar> {
        self.to_scalar(partial)
    }
}

/// Probe scalar function: test one `i64` literal against each binary bloom state.
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
        let mut set_bits = 0u64;
        let mut valid_zones = 0usize;
        for (idx, is_valid) in valid.iter().enumerate() {
            if is_valid {
                let filter = filters.bytes_at(idx);
                vortex_ensure!(
                    filter.len() == options.bytes.get(),
                    "stored bloom byte length does not match options"
                );
                set_bits += filter
                    .as_slice()
                    .iter()
                    .map(|byte| u64::from(byte.count_ones()))
                    .sum::<u64>();
                valid_zones += 1;
                possible.push(bloom_contains(
                    filter.as_slice(),
                    needle,
                    options.hashes.get(),
                ));
            } else {
                possible.push(false);
            }
        }
        if diagnostics_enabled() && valid_zones > 0 {
            let total_bits = valid_zones as f64 * options.bytes.get() as f64 * 8.0;
            let possible_zones = possible.iter().filter(|value| **value).count();
            tracing::info!(
                target: "vortex_layout::skip_index",
                index = "bloom",
                needle,
                zones = valid_zones,
                definitely_absent_zones = valid_zones - possible_zones,
                average_fill_ratio = set_bits as f64 / total_bits,
                "skip-index probe"
            );
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

/// Equality rewrite that turns a bloom miss into a zone falsifier.
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

fn i64_value(scalar: &Scalar) -> VortexResult<Option<i64>> {
    if scalar.is_null() {
        return Ok(None);
    }
    scalar
        .as_primitive_opt()
        .and_then(|primitive| primitive.typed_value::<i64>())
        .map(Some)
        .ok_or_else(|| vortex_err!("bloom value must be i64"))
}

fn bloom_insert(bits: &mut [u8], value: i64, hashes: u8) {
    bloom_insert_hash(
        bits,
        splitmix64(value as u64 ^ 0x243f_6a88_85a3_08d3),
        hashes,
    );
}

fn bloom_insert_hash(bits: &mut [u8], hash: u64, hashes: u8) {
    for (byte, bit) in bloom_positions(hash, bits.len(), hashes) {
        bits[byte] |= 1 << bit;
    }
}

fn bloom_contains(bits: &[u8], value: i64, hashes: u8) -> bool {
    bloom_contains_hash(
        bits,
        splitmix64(value as u64 ^ 0x243f_6a88_85a3_08d3),
        hashes,
    )
}

fn bloom_contains_hash(bits: &[u8], hash: u64, hashes: u8) -> bool {
    bloom_positions(hash, bits.len(), hashes).all(|(byte, bit)| bits[byte] & (1 << bit) != 0)
}

fn bloom_positions(hash: u64, bytes: usize, hashes: u8) -> impl Iterator<Item = (usize, u32)> {
    let h1 = hash;
    let h2 = splitmix64(h1 ^ 0x1319_8a2e_0370_7344) | 1;
    let bit_len = u64::try_from(bytes).unwrap_or(u64::MAX).saturating_mul(8);
    (0..u64::from(hashes)).map(move |probe| {
        let position = h1
            .wrapping_add(probe.wrapping_mul(h2))
            .wrapping_rem(bit_len);
        // `position / 8` is less than `bytes`, which is already a `usize`.
        let byte = usize::try_from(position / 8).unwrap_or_default();
        let bit = u32::try_from(position % 8).unwrap_or_default();
        (byte, bit)
    })
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn diagnostics_enabled() -> bool {
    std::env::var("VORTEX_SKIP_INDEX_DIAGNOSTICS").is_ok_and(|value| value == "1")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::Accumulator;
    use vortex_array::aggregate_fn::DynAccumulator;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::Nullability;
    use vortex_array::expr::eq;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;

    use super::*;
    use crate::layouts::zoned::zone_map::ZoneMap;

    fn small_options() -> BloomOptions {
        BloomOptions::new(
            NonZeroUsize::new(64).expect("64 is non-zero"),
            NonZeroU8::new(3).expect("3 is non-zero"),
        )
    }

    #[test]
    fn aggregate_roundtrips_options_and_membership() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let options = small_options();
        let metadata = BloomFilter
            .serialize(&options)?
            .expect("bloom is serializable");
        assert_eq!(BloomFilter.deserialize(&metadata, &session)?, options);

        let mut ctx = session.create_execution_ctx();
        let mut accumulator = Accumulator::try_new(
            BloomFilter,
            options.clone(),
            DType::Primitive(PType::I64, Nullability::NonNullable),
        )?;
        accumulator.accumulate(
            &PrimitiveArray::from_iter([10i64, 20, 30]).into_array(),
            &mut ctx,
        )?;
        let state = accumulator.finish()?;
        let bytes = state.as_binary().value().expect("bloom state is non-null");
        assert!(bloom_contains(bytes.as_slice(), 10, options.hashes.get()));
        assert!(bloom_contains(bytes.as_slice(), 20, options.hashes.get()));
        assert!(!bloom_contains(bytes.as_slice(), 999, options.hashes.get()));
        Ok(())
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
