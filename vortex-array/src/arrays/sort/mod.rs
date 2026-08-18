// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`Sort`] encoding: a lazy, stable permutation of a child array's rows.
//!
//! A `SortArray` holds a `child` array and a non-nullable `u32` `permutation` of the child's rows,
//! both as child slots. Its length equals the child's length, and output row `i` is
//! `child[permutation[i]]`. This is the raw material for sorted reads: downstream operators
//! (`take`, `filter`, aggregate, scan) read through the permutation, and executing the array (to
//! [`Canonical`]) materializes the permuted rows via a gather.
//!
//! The permutation is a full permutation (its values are a bijection `0..child.len()`), so the
//! array's length and dtype are inherited from the child. Nullability is likewise inherited; the
//! validity of the sorted array is the child's validity gathered at the permutation positions.

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::AnyCanonical;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::Canonical;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::EmptyArrayData;
use crate::array::OperationsVTable;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::array::with_empty_buffers;
use crate::array_slots;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::require_child;
use crate::scalar::Scalar;
use crate::serde::ArrayChildren;
use crate::validity::Validity;

/// A [`Sort`]-encoded Vortex array. See the [module docs](self) for the specification.
pub type SortArray = Array<Sort>;

/// The [`Sort`] encoding. See the [module docs](self).
#[derive(Clone, Debug)]
pub struct Sort;

/// Slot layout: `child` is the source array, `permutation` is the non-nullable `u32` row
/// permutation applied to it.
#[array_slots(Sort)]
pub struct SortSlots {
    /// The source array being sorted.
    #[slot(0)]
    pub child: ArrayRef,
    /// Non-nullable `u32` permutation of the child's rows.
    #[slot(1)]
    pub permutation: ArrayRef,
}

impl Sort {
    /// Validate a `(child, permutation)` pair against the [spec](self) and return the output dtype.
    fn check(child: &ArrayRef, permutation: &ArrayRef) -> VortexResult<DType> {
        // The permutation must be a non-nullable unsigned integer (u32 row indices).
        match permutation.dtype() {
            DType::Primitive(ptype, nullability) if *ptype == PType::U32 => {
                vortex_ensure!(
                    !nullability.is_nullable(),
                    "sort permutation must be non-nullable, got {}",
                    permutation.dtype()
                );
            }
            other => vortex_bail!(
                "sort permutation must be a non-nullable u32 array, got {other}"
            ),
        }

        vortex_ensure!(
            child.len() == permutation.len(),
            "sort child length {} must equal permutation length {}",
            child.len(),
            permutation.len()
        );

        let dtype = child.dtype().clone();
        Ok(dtype)
    }
}

impl Array<Sort> {
    /// Constructs a new [`SortArray`] from a `child` array and a `u32` `permutation` of the child's
    /// rows.
    ///
    /// The permutation must be non-nullable `u32`, one entry per child row. Its values must be a
    /// full permutation of `0..child.len()` — a bijection over the child's rows. The resulting
    /// array has the child's dtype and length; output row `i` is `child[permutation[i]]`.
    pub fn try_new(child: ArrayRef, permutation: ArrayRef) -> VortexResult<Self> {
        let dtype = Sort::check(&child, &permutation)?;
        // SAFETY: `check` validated the child/permutation invariants and computed the dtype.
        Ok(unsafe { Self::new_unchecked(child, permutation, dtype) })
    }

    /// Constructs a [`SortArray`] without re-validating the spec invariants.
    ///
    /// # Safety
    ///
    /// The caller must uphold every [module invariant](self): `permutation` is a non-nullable
    /// `u32` array of length `child.len()` whose values are a bijection over `0..child.len()`,
    /// and `dtype` equals the child's dtype.
    pub unsafe fn new_unchecked(child: ArrayRef, permutation: ArrayRef, dtype: DType) -> Self {
        unsafe {
            Array::from_parts_unchecked(
                ArrayParts::new(Sort, dtype, child.len(), EmptyArrayData)
                    .with_slots(
                        SortSlots {
                            child,
                            permutation,
                        }
                        .into_slots(),
                    ),
            )
        }
    }
}

impl VTable for Sort {
    type TypedArrayData = EmptyArrayData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.sort");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            slots[SortSlots::CHILD].is_some() && slots[SortSlots::PERMUTATION].is_some(),
            "SortArray requires both child and permutation slots"
        );
        let child = slots[SortSlots::CHILD]
            .as_ref()
            .vortex_expect("validated child slot");
        let permutation = slots[SortSlots::PERMUTATION]
            .as_ref()
            .vortex_expect("validated permutation slot");

        let expected_dtype = Sort::check(child, permutation)?;
        vortex_ensure!(
            dtype == &expected_dtype,
            "SortArray dtype {} does not match the dtype implied by its child {}",
            dtype,
            expected_dtype
        );
        vortex_ensure!(
            len == child.len(),
            "SortArray length {} does not match child length {}",
            len,
            child.len()
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, _idx: usize) -> BufferHandle {
        vortex_panic!("SortArray has no buffers")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SortSlots::NAMES[idx].to_string()
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        vortex_bail!("Sort array is not serializable")
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],

        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("Sort array is not serializable")
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        // Reading through the permutation is a gather: `child.take(permutation)`. The take kernel
        // is exposed via the lazy `DictArray` compute path, which requires the child to be
        // canonical form first.
        let array = require_child!(array, array.child(), SortSlots::CHILD => AnyCanonical);
        let permutation = array.permutation().clone();
        let taken = array.child().take(permutation)?;
        Ok(ExecutionResult::done(taken.execute::<Canonical>(ctx)?))
    }
}

impl OperationsVTable<Sort> for Sort {
    fn scalar_at(
        array: ArrayView<'_, Sort>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let source_row = array
            .permutation()
            .execute_scalar(index, ctx)?
            .as_primitive()
            .as_::<usize>()
            .vortex_expect("sort permutation is non-nullable u32");
        array.child().execute_scalar(source_row, ctx)
    }
}

impl ValidityVTable<Sort> for Sort {
    fn validity(array: ArrayView<'_, Sort>) -> VortexResult<Validity> {
        array.child().validity()?.take(array.permutation())
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use super::*;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;

    #[test]
    fn sort_materializes_via_permutation() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let child = buffer![5i32, 3, 9, 1, 7].into_array();
        let permutation = buffer![3u32, 1, 4, 0, 2].into_array();
        let sorted = SortArray::try_new(child, permutation)?.into_array();

        assert_arrays_eq!(
            sorted,
            buffer![1i32, 3, 7, 5, 9],
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn sort_scalar_at_reads_through_permutation() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let child = buffer![5i32, 3, 9, 1, 7].into_array();
        let permutation = buffer![3u32, 1, 4, 0, 2].into_array();
        let sorted = SortArray::try_new(child, permutation)?.into_array();

        let scalar = sorted.execute_scalar(0, &mut ctx)?;
        assert_eq!(scalar.as_primitive().typed_value::<i32>(), Some(1));
        let scalar = sorted.execute_scalar(1, &mut ctx)?;
        assert_eq!(scalar.as_primitive().typed_value::<i32>(), Some(3));
        let scalar = sorted.execute_scalar(4, &mut ctx)?;
        assert_eq!(scalar.as_primitive().typed_value::<i32>(), Some(9));
        Ok(())
    }

    #[test]
    fn sort_validity_gathered() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        // rows: [5, null, 9, 1, null]; permutation picks [1, 3, 4, 0, 2]
        let child =
            PrimitiveArray::from_option_iter([Some(5i32), None, Some(9), Some(1), None])
                .into_array();
        let permutation = buffer![1u32, 3, 4, 0, 2].into_array();
        let sorted = SortArray::try_new(child, permutation)?.into_array();

        let expected =
            PrimitiveArray::from_option_iter([None, Some(1), None, Some(5), Some(9)]).into_array();
        assert_arrays_eq!(sorted, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn sort_rejects_non_u32_permutation() {
        let child = buffer![1i32, 2].into_array();
        let bad_permutation = buffer![0u64, 1].into_array();
        let err = SortArray::try_new(child, bad_permutation)
            .err()
            .vortex_expect("expected u32 permutation requirement to reject u64");
        assert!(err.to_string().contains("u32"), "{err}");
    }

    #[test]
    fn sort_rejects_nullable_permutation() {
        use crate::arrays::PrimitiveArray;
        let child = buffer![1i32, 2].into_array();
        let bad_permutation =
            PrimitiveArray::from_option_iter([Some(0u32), Some(1)]).into_array();
        let err = SortArray::try_new(child, bad_permutation)
            .err()
            .vortex_expect("expected non-nullable permutation requirement to reject nullable");
        assert!(err.to_string().contains("non-nullable"), "{err}");
    }

    #[test]
    fn sort_rejects_length_mismatch() {
        let child = buffer![1i32, 2].into_array();
        let bad_permutation = buffer![0u32].into_array();
        let err = SortArray::try_new(child, bad_permutation)
            .err()
            .vortex_expect("expected length mismatch to be rejected");
        assert!(err.to_string().contains("length"), "{err}");
    }
}
