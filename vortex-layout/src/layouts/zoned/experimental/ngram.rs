// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::num::NonZeroU8;

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
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::literal::Literal;
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

use super::BloomOptions;
use super::BloomPartial;
use super::SkipIndex;
use super::bloom_contains_bytes;
use super::bloom_insert_bytes;

/// Persisted tuning for a bloom filter populated with every byte n-gram in a UTF-8 zone.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NGramBloomOptions {
    bloom: BloomOptions,
    gram_size: NonZeroU8,
}

impl NGramBloomOptions {
    /// Create n-gram bloom options.
    pub fn new(bloom: BloomOptions, gram_size: NonZeroU8) -> Self {
        Self { bloom, gram_size }
    }

    /// Bloom sizing and hash-count options.
    pub fn bloom(&self) -> &BloomOptions {
        &self.bloom
    }

    /// Number of UTF-8 bytes in each indexed gram.
    pub fn gram_size(&self) -> NonZeroU8 {
        self.gram_size
    }
}

impl Default for NGramBloomOptions {
    fn default() -> Self {
        Self {
            bloom: BloomOptions::new(
                std::num::NonZeroUsize::new(64 * 1024).unwrap_or(std::num::NonZeroUsize::MIN),
                NonZeroU8::new(5).unwrap_or(NonZeroU8::MIN),
            ),
            gram_size: NonZeroU8::new(3).unwrap_or(NonZeroU8::MIN),
        }
    }
}

impl Display for NGramBloomOptions {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{},gram_size={}", self.bloom, self.gram_size)
    }
}

/// Experimental byte n-gram bloom index for case-sensitive SQL `LIKE` predicates.
#[derive(Clone, Debug, Default)]
pub struct NGramBloomSkipIndex {
    options: NGramBloomOptions,
}

impl NGramBloomSkipIndex {
    /// Create an index with explicit bloom and gram tuning.
    pub fn new(options: NGramBloomOptions) -> Self {
        Self { options }
    }

    /// Persisted index options.
    pub fn options(&self) -> &NGramBloomOptions {
        &self.options
    }
}

impl SkipIndex for NGramBloomSkipIndex {
    fn aggregate_fn(&self, input_dtype: &DType) -> Option<AggregateFnRef> {
        NGramBloomFilter
            .return_dtype(&self.options, input_dtype)
            .map(|_| NGramBloomFilter.bind(self.options.clone()))
    }

    fn register(&self, session: &VortexSession) {
        session.aggregate_fns().register(NGramBloomFilter);
        session.scalar_fns().register(NGramBloomContains);
        session.stats().register_rewrite(NGramLikeRewrite {
            options: self.options.clone(),
        });
    }
}

#[derive(Clone, Debug)]
struct NGramBloomFilter;

impl AggregateFnVTable for NGramBloomFilter {
    type Options = NGramBloomOptions;
    type Partial = BloomPartial;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.experimental.ngram_bloom.utf8.v1");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        let bytes = u32::try_from(options.bloom.bytes().get())?;
        let mut metadata = bytes.to_le_bytes().to_vec();
        metadata.push(options.bloom.hashes().get());
        metadata.push(options.gram_size.get());
        Ok(Some(metadata))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        vortex_ensure!(metadata.len() == 6, "invalid n-gram bloom metadata length");
        let bytes = u32::from_le_bytes([metadata[0], metadata[1], metadata[2], metadata[3]]);
        Ok(NGramBloomOptions::new(
            BloomOptions::new(
                std::num::NonZeroUsize::new(usize::try_from(bytes)?)
                    .ok_or_else(|| vortex_err!("n-gram bloom byte length must be non-zero"))?,
                NonZeroU8::new(metadata[4])
                    .ok_or_else(|| vortex_err!("n-gram bloom hash count must be non-zero"))?,
            ),
            NonZeroU8::new(metadata[5])
                .ok_or_else(|| vortex_err!("n-gram size must be non-zero"))?,
        ))
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        matches!(input_dtype, DType::Utf8(_)).then_some(DType::Binary(Nullability::NonNullable))
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
            bits: vec![0; options.bloom.bytes().get()],
            hashes: options.bloom.hashes().get(),
            gram_size: Some(options.gram_size.get()),
        })
    }

    fn combine_partials(&self, partial: &mut Self::Partial, other: Scalar) -> VortexResult<()> {
        if other.is_null() {
            return Ok(());
        }
        let other = other
            .as_binary()
            .value()
            .ok_or_else(|| vortex_err!("non-null n-gram bloom partial has no bytes"))?;
        vortex_ensure!(
            partial.bits.len() == other.len(),
            "n-gram bloom partial length mismatch"
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
        let gram_size = usize::from(
            partial
                .gram_size
                .ok_or_else(|| vortex_err!("n-gram bloom partial is missing its gram size"))?,
        );
        match batch {
            Columnar::Constant(constant) => {
                if let Some(value) = constant.scalar().as_utf8().value() {
                    insert_grams(
                        &mut partial.bits,
                        value.as_bytes(),
                        gram_size,
                        partial.hashes,
                    );
                }
            }
            Columnar::Canonical(canonical) => {
                let values = canonical.as_varbinview();
                let validity = values
                    .varbinview_validity()
                    .execute_mask(values.len(), ctx)?;
                for (idx, valid) in validity.iter().enumerate() {
                    if valid {
                        insert_grams(
                            &mut partial.bits,
                            values.bytes_at(idx).as_slice(),
                            gram_size,
                            partial.hashes,
                        );
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

#[derive(Clone, Debug)]
struct NGramBloomContains;

impl ScalarFnVTable for NGramBloomContains {
    type Options = NGramBloomOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.experimental.ngram_bloom_contains.utf8.v1");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        NGramBloomFilter.serialize(options)
    }

    fn deserialize(&self, metadata: &[u8], session: &VortexSession) -> VortexResult<Self::Options> {
        NGramBloomFilter.deserialize(metadata, session)
    }

    fn arity(&self, _options: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _options: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("filter"),
            1 => ChildName::from("pattern"),
            _ => unreachable!("ngram_bloom_contains has exactly two children"),
        }
    }

    fn return_dtype(&self, _options: &Self::Options, args: &[DType]) -> VortexResult<DType> {
        vortex_ensure!(
            matches!(args[0], DType::Binary(_)),
            "n-gram bloom must be Binary"
        );
        vortex_ensure!(
            matches!(args[1], DType::Utf8(_)),
            "LIKE pattern must be Utf8"
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
        let pattern = args
            .get(1)?
            .as_constant()
            .ok_or_else(|| vortex_err!("LIKE pattern must be constant"))?;
        let Some(pattern) = pattern.as_utf8().value() else {
            return Ok(ConstantArray::new(
                Scalar::null(DType::Bool(Nullability::Nullable)),
                args.row_count(),
            )
            .into_array());
        };
        let grams = required_grams(pattern, usize::from(options.gram_size.get()));

        let validity = filters.varbinview_validity();
        let valid = validity.execute_mask(filters.len(), ctx)?;
        let mut possible = Vec::with_capacity(filters.len());
        let mut set_bits = 0u64;
        let mut valid_zones = 0usize;
        for (idx, is_valid) in valid.iter().enumerate() {
            if !is_valid {
                possible.push(false);
                continue;
            }
            let filter = filters.bytes_at(idx);
            vortex_ensure!(
                filter.len() == options.bloom.bytes().get(),
                "stored n-gram bloom byte length does not match options"
            );
            set_bits += filter
                .as_slice()
                .iter()
                .map(|byte| u64::from(byte.count_ones()))
                .sum::<u64>();
            valid_zones += 1;
            possible.push(grams.iter().all(|gram| {
                bloom_contains_bytes(filter.as_slice(), gram, options.bloom.hashes().get())
            }));
        }

        if diagnostics_enabled() && valid_zones > 0 {
            let total_bits = valid_zones as f64 * options.bloom.bytes().get() as f64 * 8.0;
            let possible_zones = possible.iter().filter(|value| **value).count();
            tracing::info!(
                target: "vortex_layout::skip_index",
                index = "ngram_bloom",
                pattern = pattern.as_str(),
                gram_size = options.gram_size.get(),
                grams = grams.len(),
                zones = valid_zones,
                definitely_absent_zones = valid_zones - possible_zones,
                average_fill_ratio = set_bits as f64 / total_bits,
                "experimental skip-index probe"
            );
        }

        Ok(BoolArray::new(BitBuffer::from_iter(possible), validity).into_array())
    }

    fn is_null_sensitive(&self, _options: &Self::Options) -> bool {
        false
    }

    fn is_fallible(&self, _options: &Self::Options) -> bool {
        true
    }
}

#[derive(Clone, Debug)]
struct NGramLikeRewrite {
    options: NGramBloomOptions,
}

impl StatsRewriteRule for NGramLikeRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        Like.id()
    }

    fn falsify(
        &self,
        expr: &Expression,
        ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>> {
        let like = expr.as_::<Like>();
        if like.negated || like.case_insensitive || !is_root(expr.child(0)) {
            return Ok(None);
        }
        let Some(pattern) = expr.child(1).as_opt::<Literal>() else {
            return Ok(None);
        };
        let Some(pattern_value) = pattern.as_utf8().value() else {
            return Ok(None);
        };
        if !matches!(ctx.return_dtype(expr.child(0))?, DType::Utf8(_))
            || required_grams(pattern_value, usize::from(self.options.gram_size.get())).is_empty()
        {
            return Ok(None);
        }

        let filter = stat(
            expr.child(0).clone(),
            NGramBloomFilter.bind(self.options.clone()),
        );
        let contains =
            NGramBloomContains.new_expr(self.options.clone(), [filter, expr.child(1).clone()]);
        Ok(Some(not(contains)))
    }
}

fn insert_grams(bits: &mut [u8], value: &[u8], gram_size: usize, hashes: u8) {
    for gram in value.windows(gram_size) {
        bloom_insert_bytes(bits, gram, hashes);
    }
}

fn required_grams(pattern: &str, gram_size: usize) -> Vec<Vec<u8>> {
    literal_runs(pattern)
        .into_iter()
        .flat_map(|run| {
            run.windows(gram_size)
                .map(<[u8]>::to_vec)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn literal_runs(pattern: &str) -> Vec<Vec<u8>> {
    let mut runs = Vec::new();
    let mut current = Vec::new();
    let mut chars = pattern.chars();
    while let Some(character) = chars.next() {
        match character {
            '\\' => {
                if let Some(escaped) = chars.next() {
                    let mut bytes = [0; 4];
                    current.extend_from_slice(escaped.encode_utf8(&mut bytes).as_bytes());
                } else {
                    current.push(b'\\');
                }
            }
            '%' | '_' => {
                if !current.is_empty() {
                    runs.push(std::mem::take(&mut current));
                }
            }
            character => {
                let mut bytes = [0; 4];
                current.extend_from_slice(character.encode_utf8(&mut bytes).as_bytes());
            }
        }
    }
    if !current.is_empty() {
        runs.push(current);
    }
    runs
}

fn diagnostics_enabled() -> bool {
    std::env::var("VORTEX_EXPERIMENTAL_SKIP_INDEX_DIAGNOSTICS").is_ok_and(|value| value == "1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_only_required_literal_grams() {
        assert_eq!(
            required_grams(r"%foo_bar\%baz%", 3),
            [
                b"foo".to_vec(),
                b"bar".to_vec(),
                b"ar%".to_vec(),
                b"r%b".to_vec(),
                b"%ba".to_vec(),
                b"baz".to_vec()
            ]
        );
        assert!(required_grams("%a%b%", 3).is_empty());
    }

    #[test]
    fn gram_membership_has_no_false_negatives() {
        let options = NGramBloomOptions::default();
        let mut bits = vec![0; options.bloom.bytes().get()];
        insert_grams(
            &mut bits,
            b"the quick brown fox",
            usize::from(options.gram_size.get()),
            options.bloom.hashes().get(),
        );
        for gram in b"quick".windows(3) {
            assert!(bloom_contains_bytes(
                &bits,
                gram,
                options.bloom.hashes().get()
            ));
        }
    }
}
