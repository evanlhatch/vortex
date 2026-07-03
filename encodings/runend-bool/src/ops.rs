// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::search_sorted::SearchResult;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedPrimitiveArray;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;

use crate::RunEndBool;
use crate::array::RunEndBoolArrayExt;
use crate::compress::value_at_index;

impl OperationsVTable<RunEndBool> for RunEndBool {
    fn scalar_at(
        array: ArrayView<'_, RunEndBool>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // Honor validity: a null logical element produces a null scalar.
        let validity_mask = array
            .bool_validity()
            .execute_mask(array.as_ref().len(), ctx)?;
        if !validity_mask.value(index) {
            return Ok(Scalar::null(array.as_ref().dtype().clone()));
        }

        let run_index = array.find_physical_index(index, ctx)?;
        let value = value_at_index(run_index, array.start());
        Ok(Scalar::bool(value, array.nullability()))
    }
}

/// Find the physical run index containing logical `index` for the given `ends` child.
///
/// The caller is responsible for adding any run `offset` to `index` before calling.
pub(crate) fn find_physical_index(
    array: &ArrayRef,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<usize> {
    match_each_unsigned_integer_ptype!(array.dtype().as_ptype(), |T| {
        Ok(SearchSortedPrimitiveArray::<T>::new(array, ctx)
            .search_sorted(&index, SearchSortedSide::Right)?
            .to_ends_index(array.len()))
    })
}

/// Find the physical offset for an index that would be an end of the slice i.e., one past the last element.
///
/// If the index exists in the array we want to take that position (as we are searching from the right)
/// otherwise we want to take the next one.
pub(crate) fn find_slice_end_index(
    array: &ArrayRef,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<usize> {
    let result = match_each_unsigned_integer_ptype!(array.dtype().as_ptype(), |T| {
        SearchSortedPrimitiveArray::<T>::new(array, ctx)
            .search_sorted(&index, SearchSortedSide::Right)?
    });
    Ok(match result {
        SearchResult::Found(i) => i,
        SearchResult::NotFound(i) => {
            if i == array.len() {
                i
            } else {
                i + 1
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::assert_arrays_eq;
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
    fn slice_array() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // [T,T,F,F,F,T,T,T,T,T] sliced 3..8 => [F,F,T,T,T]
        let arr = RunEndBool::try_new(
            buffer![2u32, 5, 10].into_array(),
            true,
            Validity::NonNullable,
            &mut ctx,
        )?
        .slice(3..8)?;
        assert_eq!(arr.len(), 5);
        let expected = BoolArray::from(BitBuffer::from(vec![false, false, true, true, true]));
        assert_arrays_eq!(arr, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn double_slice() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = RunEndBool::try_new(
            buffer![2u32, 5, 10].into_array(),
            true,
            Validity::NonNullable,
            &mut ctx,
        )?
        .slice(3..8)?;
        let doubly_sliced = arr.slice(0..3)?;
        let expected = BoolArray::from(BitBuffer::from(vec![false, false, true]));
        assert_arrays_eq!(doubly_sliced, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn slice_to_empty() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let arr = RunEndBool::try_new(
            buffer![2u32, 5, 10].into_array(),
            true,
            Validity::NonNullable,
            &mut ctx,
        )?;
        let sliced = arr.slice(arr.len()..arr.len())?;
        assert!(sliced.is_empty());
        Ok(())
    }
}
