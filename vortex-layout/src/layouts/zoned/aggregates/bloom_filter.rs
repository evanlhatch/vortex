//! Bloom-filter aggregate for zoned layouts.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::num::NonZeroU8;
use std::num::NonZeroUsize;

use vortex_array::ArrayRef;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnId;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

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

/// Aggregate that stores one fixed-size Bloom bitset as a `Binary` scalar for every zone.
#[derive(Clone, Debug)]
pub(in crate::layouts::zoned) struct BloomFilter;

/// In-memory Bloom accumulator. Only the bitset is persisted.
pub(in crate::layouts::zoned) struct BloomPartial {
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

pub(in crate::layouts::zoned) fn i64_value(scalar: &Scalar) -> VortexResult<Option<i64>> {
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

pub(in crate::layouts::zoned) fn bloom_contains(bits: &[u8], value: i64, hashes: u8) -> bool {
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

#[cfg(test)]
mod tests {
    use std::num::NonZeroU8;
    use std::num::NonZeroUsize;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::Accumulator;
    use vortex_array::aggregate_fn::AggregateFnVTable;
    use vortex_array::aggregate_fn::DynAccumulator;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_error::VortexResult;

    use super::BloomFilter;
    use super::BloomOptions;
    use super::bloom_contains;

    fn small_options() -> BloomOptions {
        BloomOptions::new(
            NonZeroUsize::new(64).expect("64 is non-zero"),
            NonZeroU8::new(3).expect("3 is non-zero"),
        )
    }

    #[test]
    fn roundtrips_options_and_membership() -> VortexResult<()> {
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
}
