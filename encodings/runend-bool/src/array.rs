// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::TypedArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Bool;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::bool::BoolArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::child_to_validity;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::legacy_session;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::serde::ArrayChildren;
use vortex_array::smallvec::smallvec;
use vortex_array::validity::Validity;
use vortex_array::validity_to_child;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_runend::trimmed_ends_iter;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::compress::encode_runend_bool;
use crate::compress::runend_bool_decode_slice;

/// A [`RunEndBool`]-encoded Vortex array.
pub type RunEndBoolArray = Array<RunEndBool>;

#[derive(Clone, prost::Message)]
pub struct RunEndBoolMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    pub ends_ptype: i32,
    #[prost(uint64, tag = "2")]
    pub num_runs: u64,
    #[prost(uint64, tag = "3")]
    pub offset: u64,
    #[prost(bool, tag = "4")]
    pub start: bool,
}

impl ArrayHash for RunEndBoolData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.offset.hash(state);
        self.start.hash(state);
    }
}

impl ArrayEq for RunEndBoolData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.offset == other.offset && self.start == other.start
    }
}

impl VTable for RunEndBool {
    type TypedArrayData = RunEndBoolData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.runend_bool");
        *ID
    }

    #[allow(clippy::disallowed_methods)]
    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let DType::Bool(nullability) = dtype else {
            vortex_bail!("Expected bool dtype, got {dtype:?}");
        };
        let ends = slots[ENDS_SLOT]
            .as_ref()
            .vortex_expect("RunEndBoolArray ends slot");
        // TODO(ctx): trait fixes - VTable::validate has a fixed signature.
        let mut ctx = legacy_session().create_execution_ctx();
        RunEndBoolData::validate_parts(ends, data.offset, len, &mut ctx)?;

        let validity = child_to_validity(slots[VALIDITY_SLOT].as_ref(), *nullability);
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == len,
                "RunEndBoolArray validity len {} does not match outer length {}",
                validity_len,
                len
            );
        }
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("RunEndBoolArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("RunEndBoolArray buffer_name index {idx} out of bounds")
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_array::vtable::with_empty_buffers(self, array, buffers)
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(
            RunEndBoolMetadata {
                ends_ptype: PType::try_from(array.ends().dtype())
                    .vortex_expect("Must be a valid PType") as i32,
                num_runs: array.ends().len() as u64,
                offset: array.offset() as u64,
                start: array.start(),
            }
            .encode_to_vec(),
        ))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = RunEndBoolMetadata::decode(metadata)?;
        let ends_dtype = DType::Primitive(metadata.ends_ptype(), Nullability::NonNullable);
        let runs = usize::try_from(metadata.num_runs).vortex_expect("Must be a valid usize");
        let ends = children.get(0, &ends_dtype, runs)?;

        // Validity is an optional child whose index depends on whether ends consumed a slot.
        let validity = if children.len() <= 1 {
            Validity::from(dtype.nullability())
        } else {
            Validity::Array(children.get(1, &Validity::DTYPE, len)?)
        };

        let offset = usize::try_from(metadata.offset).vortex_expect("Offset must be a valid usize");
        let slots = smallvec![Some(ends), validity_to_child(&validity, len)];
        let data = RunEndBoolData::new(offset, metadata.start);
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        run_end_bool_canonicalize(&array, ctx).map(ExecutionResult::done)
    }
}

/// The run-end positions marking where each run terminates.
pub(super) const ENDS_SLOT: usize = 0;
/// The optional validity child.
pub(super) const VALIDITY_SLOT: usize = 1;
pub(super) const NUM_SLOTS: usize = 2;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["ends", "validity"];

#[derive(Clone, Debug)]
pub struct RunEndBoolData {
    offset: usize,
    start: bool,
}

impl Display for RunEndBoolData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "offset: {}, start: {}", self.offset, self.start)
    }
}

/// Extension methods for [`RunEndBoolArray`].
pub trait RunEndBoolArrayExt: TypedArrayRef<RunEndBool> {
    /// The logical offset into the first run.
    fn offset(&self) -> usize {
        self.offset
    }

    /// The boolean value of the first (run index 0) run.
    fn start(&self) -> bool {
        self.start
    }

    /// The primitive array of strictly-increasing run end positions.
    fn ends(&self) -> &ArrayRef {
        self.as_ref().slots()[ENDS_SLOT]
            .as_ref()
            .vortex_expect("RunEndBoolArray ends slot")
    }

    /// The array's validity.
    fn bool_validity(&self) -> Validity {
        child_to_validity(
            self.as_ref().slots()[VALIDITY_SLOT].as_ref(),
            self.nullability(),
        )
    }

    /// The array's nullability.
    fn nullability(&self) -> Nullability {
        match self.as_ref().dtype() {
            DType::Bool(nullability) => *nullability,
            _ => unreachable!("RunEndBoolArray requires a bool dtype"),
        }
    }

    /// Find the physical run index containing the given logical `index`.
    fn find_physical_index(&self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<usize> {
        crate::ops::find_physical_index(self.ends(), index + self.offset(), ctx)
    }
}
impl<T: TypedArrayRef<RunEndBool>> RunEndBoolArrayExt for T {}

#[derive(Clone, Debug)]
pub struct RunEndBool;

impl RunEndBool {
    /// Build a new [`RunEndBoolArray`] without validation.
    ///
    /// # Safety
    /// The caller must ensure `ends` are strictly increasing unsigned integers and that the last
    /// run end is `>= offset + length`.
    pub unsafe fn new_unchecked(
        ends: ArrayRef,
        start: bool,
        offset: usize,
        length: usize,
        validity: Validity,
    ) -> RunEndBoolArray {
        let dtype = DType::Bool(validity.nullability());
        let slots = smallvec![Some(ends), validity_to_child(&validity, length)];
        let data = unsafe { RunEndBoolData::new_unchecked(offset, start) };
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(RunEndBool, dtype, length, data).with_slots(slots),
            )
        }
    }

    /// Build a new [`RunEndBoolArray`] from ends, start, and validity.
    pub fn try_new(
        ends: ArrayRef,
        start: bool,
        validity: Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<RunEndBoolArray> {
        let len = RunEndBoolData::logical_len_from_ends(&ends, ctx)?;
        Self::try_new_offset_length(ends, start, 0, len, validity, ctx)
    }

    /// Build a new [`RunEndBoolArray`] from ends, start, offset, length, and validity.
    pub fn try_new_offset_length(
        ends: ArrayRef,
        start: bool,
        offset: usize,
        length: usize,
        validity: Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<RunEndBoolArray> {
        RunEndBoolData::validate_parts(&ends, offset, length, ctx)?;
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == length,
                "validity len {validity_len} does not match length {length}"
            );
        }
        let dtype = DType::Bool(validity.nullability());
        let slots = smallvec![Some(ends), validity_to_child(&validity, length)];
        let data = RunEndBoolData::new(offset, start);
        Array::try_from_parts(ArrayParts::new(RunEndBool, dtype, length, data).with_slots(slots))
    }

    /// Build a new [`RunEndBoolArray`] from ends, start, and validity (panics on invalid input).
    ///
    /// # Examples
    ///
    /// ```
    /// # use vortex_array::IntoArray;
    /// # use vortex_array::VortexSessionExecute;
    /// # use vortex_array::validity::Validity;
    /// # use vortex_buffer::buffer;
    /// # use vortex_error::VortexResult;
    /// # use vortex_runend_bool::RunEndBool;
    /// # fn main() -> VortexResult<()> {
    /// let session = vortex_array::array_session();
    /// vortex_runend_bool::initialize(&session);
    /// let mut ctx = session.create_execution_ctx();
    /// // start = false, so runs alternate false, true: [false, false, true]
    /// let ends = buffer![2u8, 3u8].into_array();
    /// let run_end = RunEndBool::new(ends, false, Validity::NonNullable, &mut ctx);
    ///
    /// assert_eq!(run_end.execute_scalar(0, &mut ctx)?, false.into());
    /// assert_eq!(run_end.execute_scalar(1, &mut ctx)?, false.into());
    /// assert_eq!(run_end.execute_scalar(2, &mut ctx)?, true.into());
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        ends: ArrayRef,
        start: bool,
        validity: Validity,
        ctx: &mut ExecutionCtx,
    ) -> RunEndBoolArray {
        Self::try_new(ends, start, validity, ctx).vortex_expect("RunEndBoolData is always valid")
    }

    /// Run-end encode a boolean array.
    pub fn encode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<RunEndBoolArray> {
        if let Some(barray) = array.as_opt::<Bool>() {
            encode_runend_bool(barray, ctx)
        } else {
            vortex_bail!("RunEndBool can only encode bool arrays")
        }
    }
}

impl RunEndBoolData {
    fn logical_len_from_ends(ends: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<usize> {
        if ends.is_empty() {
            Ok(0)
        } else {
            usize::try_from(&ends.execute_scalar(ends.len() - 1, ctx)?)
        }
    }

    pub(crate) fn validate_parts(
        ends: &ArrayRef,
        offset: usize,
        length: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        vortex_ensure!(
            ends.dtype().is_unsigned_int(),
            "run ends must be unsigned integers, was {}",
            ends.dtype(),
        );

        // Handle empty run-ends.
        if ends.is_empty() {
            vortex_ensure!(
                offset == 0,
                "non-zero offset provided for empty RunEndBoolArray"
            );
            return Ok(());
        }

        // Zero-length logical slices may retain run metadata from the source array.
        if length == 0 {
            return Ok(());
        }

        #[cfg(debug_assertions)]
        {
            // Run ends must be strictly sorted for binary search to work correctly.
            let pre_validation = ends.statistics().to_owned();

            let is_sorted = ends
                .statistics()
                .compute_is_strict_sorted(ctx)
                .unwrap_or(false);

            // Preserve the original statistics since compute_is_strict_sorted may have mutated them.
            ends.statistics().inherit(pre_validation.iter());
            debug_assert!(is_sorted);
        }

        // Skip host-only validation when ends are not host-resident.
        if !ends.is_host() {
            return Ok(());
        }

        // Validate the offset and length are valid for the given ends.
        if offset != 0 && length != 0 {
            let first_run_end = usize::try_from(&ends.execute_scalar(0, ctx)?)?;
            if first_run_end < offset {
                vortex_bail!("First run end {first_run_end} must be >= offset {offset}");
            }
        }

        let last_run_end = usize::try_from(&ends.execute_scalar(ends.len() - 1, ctx)?)?;
        let min_required_end = offset + length;
        if last_run_end < min_required_end {
            vortex_bail!("Last run end {last_run_end} must be >= offset+length {min_required_end}");
        }

        Ok(())
    }

    /// Build new inner data from an offset and the value of the first run.
    pub fn new(offset: usize, start: bool) -> Self {
        Self { offset, start }
    }

    /// Build new inner data without validation.
    ///
    /// # Safety
    ///
    /// See [`RunEndBool::try_new_offset_length`] for the required preconditions.
    pub unsafe fn new_unchecked(offset: usize, start: bool) -> Self {
        Self { offset, start }
    }
}

impl ValidityVTable<RunEndBool> for RunEndBool {
    fn validity(array: ArrayView<'_, RunEndBool>) -> VortexResult<Validity> {
        Ok(array.bool_validity())
    }
}

pub(super) fn run_end_bool_canonicalize(
    array: &RunEndBoolArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let ends = array.ends().clone().execute::<PrimitiveArray>(ctx)?;
    let offset = array.offset();
    let length = array.as_ref().len();
    let start = array.start();

    let bits = match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
        runend_bool_decode_slice(
            trimmed_ends_iter(ends.as_slice::<E>(), offset, length),
            start,
            length,
        )
    });

    let validity = array.bool_validity().execute_mask(length, ctx)?;
    let validity = Validity::from_mask(validity, array.nullability());

    Ok(BoolArray::new(bits, validity).into_array())
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::RunEndBool;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    #[test]
    fn test_constructor() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = RunEndBool::new(
            buffer![2u32, 5, 10].into_array(),
            true,
            Validity::NonNullable,
            &mut ctx,
        );
        assert_eq!(arr.len(), 10);
        assert_eq!(arr.dtype(), &DType::Bool(Nullability::NonNullable));

        let expected = BoolArray::from(BitBuffer::from(vec![
            true, true, false, false, false, true, true, true, true, true,
        ]));
        assert_arrays_eq!(arr.into_array(), expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_encode_roundtrip() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let bits = vec![
            true, true, false, false, false, true, true, true, true, true,
        ];
        let bool_array = BoolArray::from(BitBuffer::from(bits));
        let encoded = RunEndBool::encode(bool_array.clone().into_array(), &mut ctx)?;
        assert_arrays_eq!(encoded.into_array(), bool_array, &mut ctx);
        Ok(())
    }
}
