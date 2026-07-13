// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use futures::try_join;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::RowSplits;
use crate::SplitRange;
use crate::layouts::fixed_size_list::FixedSizeListLayout;
use crate::layouts::fixed_size_list::expr::FixedSizeListChildrenNeeded;
use crate::layouts::fixed_size_list::expr::get_necessary_fixed_size_list_children;
use crate::layouts::fixed_size_list::expr::rewrite_list_length_expr;
use crate::layouts::fixed_size_list::expr::rewrite_validity_expr;
use crate::segments::SegmentSource;

type OptionalArrayFuture = BoxFuture<'static, VortexResult<Option<ArrayRef>>>;

/// The threshold of mask density below which we push the input mask into projection evaluation,
/// and above which we evaluate the expression over all rows and intersect afterward.
const EXPR_EVAL_THRESHOLD: f64 = 0.2;

#[derive(Clone)]
pub(super) struct FixedSizeListReader {
    layout: FixedSizeListLayout,
    name: Arc<str>,
    session: VortexSession,
    elements: LayoutReaderRef,
    validity: Option<LayoutReaderRef>,
}

impl FixedSizeListReader {
    pub(super) fn try_new(
        layout: FixedSizeListLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: VortexSession,
        ctx: &LayoutReaderContext,
    ) -> VortexResult<Self> {
        let elements = layout.elements().new_reader(
            format!("{name}.elements").into(),
            Arc::clone(&segment_source),
            &session,
            ctx,
        )?;
        let validity = layout
            .validity()
            .map(|v| {
                v.new_reader(
                    format!("{name}.validity").into(),
                    Arc::clone(&segment_source),
                    &session,
                    ctx,
                )
            })
            .transpose()?;

        Ok(Self {
            layout,
            name,
            session,
            elements,
            validity,
        })
    }

    fn project_validity(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let validity_reader = self.validity.clone();
        let nullability = self.layout.dtype().nullability();
        let row_range = row_range.clone();
        let rewritten = rewrite_validity_expr(expr)?;

        Ok(async move {
            let mask = mask.await?;
            let row_count = usize::try_from(row_range.end - row_range.start)?;
            let out_len = if mask.all_true() {
                row_count
            } else {
                mask.true_count()
            };

            let validity_array = match validity_reader.as_ref() {
                Some(v) => Some(
                    v.projection_evaluation(&row_range, &root(), MaskFuture::ready(mask))?
                        .await?,
                ),
                None => None,
            };

            create_validity(validity_array, nullability)
                .to_array(out_len)
                .apply(&rewritten)
        }
        .boxed())
    }

    fn project_list_length(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let list_size = u64::from(self.layout.list_size());
        let nullability = self.layout.dtype().nullability();
        let row_count = usize::try_from(row_range.end - row_range.start)?;
        let rewritten = rewrite_list_length_expr(expr)?;
        let validity_fut = fetch_validity(
            self.validity.as_ref(),
            row_range,
            MaskFuture::new_true(row_count),
        )?;

        Ok(async move {
            let validity = validity_fut.await?;
            let lengths = ConstantArray::new(list_size, row_count)
                .into_array()
                .cast(DType::Primitive(PType::U64, nullability))?;
            let lengths = apply_validity(lengths, validity, nullability)?;

            let mask = mask.await?;
            let lengths = if mask.all_true() {
                lengths
            } else {
                lengths.filter(mask)?
            };
            lengths.apply(&rewritten)
        }
        .boxed())
    }

    fn project_elements(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let reader = self.clone();
        let expr = expr.clone();
        let row_range = row_range.clone();

        Ok(async move {
            let row_count = usize::try_from(row_range.end - row_range.start)?;
            let list_size = u64::from(reader.layout.list_size());
            let elements_range = element_range(&row_range, list_size)?;
            let elements_len = usize::try_from(elements_range.end - elements_range.start)?;

            let elements_fut = reader.elements.projection_evaluation(
                &elements_range,
                &root(),
                MaskFuture::new_true(elements_len),
            )?;

            let validity_fut = fetch_validity(
                reader.validity.as_ref(),
                &row_range,
                MaskFuture::new_true(row_count),
            )?;

            let (elements, validity) = try_join!(elements_fut, validity_fut)?;
            let fsl = build_fixed_size_list(elements, validity, reader.layout.dtype(), row_count)?;

            let mask = mask.await?;
            let fsl = if mask.all_true() {
                fsl
            } else {
                fsl.filter(mask)?
            };
            fsl.apply(&expr)
        }
        .boxed())
    }
}

impl LayoutReader for FixedSizeListReader {
    fn name(&self) -> &Arc<str> {
        &self.name
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.layout.dtype()
    }

    fn row_count(&self) -> u64 {
        self.layout.row_count()
    }

    fn register_splits(
        &self,
        field_mask: &[FieldMask],
        split_range: &SplitRange,
        splits: &mut RowSplits,
    ) -> VortexResult<()> {
        split_range.check_bounds(self.layout.row_count())?;
        splits.push(split_range.root_row_range().end);

        let list_size = u64::from(self.layout.list_size());
        if list_size != 0 {
            let element_range = element_range(split_range.row_range(), list_size)?;
            let mut element_splits = RowSplits::new_capacity(8);
            self.elements.register_splits(
                field_mask,
                &SplitRange::try_new(0, element_range.clone())?,
                &mut element_splits,
            )?;

            for element_split in element_splits.into_sorted_deduped() {
                if element_split <= element_range.start || element_split >= element_range.end {
                    continue;
                }
                let element_offset = element_split - element_range.start;
                let row_offset = element_offset.div_ceil(list_size);
                let root_row = split_range
                    .root_row_range()
                    .start
                    .checked_add(row_offset)
                    .ok_or_else(|| vortex_err!("fixed-size-list split offset overflow"))?;
                if root_row < split_range.root_row_range().end {
                    splits.push(root_row);
                }
            }
        }

        if let Some(validity) = &self.validity {
            validity.register_splits(field_mask, split_range, splits)?;
        }
        Ok(())
    }

    // TODO(mk): either have zone pruning upstream or implement here
    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        Ok(MaskFuture::ready(mask))
    }

    fn filter_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<MaskFuture> {
        let len = mask.len();
        let reader = self.clone();
        let row_range = row_range.clone();
        let expr = expr.clone();
        let session = self.session.clone();

        Ok(MaskFuture::new(len, async move {
            let mask = mask.await?;
            if mask.all_false() {
                return Ok(mask);
            }

            if mask.density() < EXPR_EVAL_THRESHOLD {
                let predicate = reader
                    .projection_evaluation(&row_range, &expr, MaskFuture::ready(mask.clone()))?
                    .await?;
                let predicate_mask = predicate_array_to_mask(predicate, &session)?;
                Ok(mask.intersect_by_rank(&predicate_mask))
            } else {
                let predicate = reader
                    .projection_evaluation(&row_range, &expr, MaskFuture::new_true(len))?
                    .await?;
                let predicate_mask = predicate_array_to_mask(predicate, &session)?;
                Ok(mask & &predicate_mask)
            }
        }))
    }

    fn projection_evaluation(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        match get_necessary_fixed_size_list_children(expr) {
            FixedSizeListChildrenNeeded::Validity => self.project_validity(row_range, expr, mask),
            FixedSizeListChildrenNeeded::ListLengthAndValidity => {
                self.project_list_length(row_range, expr, mask)
            }
            FixedSizeListChildrenNeeded::Elements => self.project_elements(row_range, expr, mask),
        }
    }
}

fn element_range(row_range: &Range<u64>, list_size: u64) -> VortexResult<Range<u64>> {
    let start = row_range
        .start
        .checked_mul(list_size)
        .ok_or_else(|| vortex_err!("fixed-size-list element range overflow"))?;
    let end = row_range
        .end
        .checked_mul(list_size)
        .ok_or_else(|| vortex_err!("fixed-size-list element range overflow"))?;
    Ok(start..end)
}

fn fetch_validity(
    validity: Option<&LayoutReaderRef>,
    row_range: &Range<u64>,
    mask: MaskFuture,
) -> VortexResult<OptionalArrayFuture> {
    let fut = validity
        .map(|v| v.projection_evaluation(row_range, &root(), mask))
        .transpose()?;
    Ok(async move {
        match fut {
            Some(f) => f.await.map(Some),
            None => Ok(None),
        }
    }
    .boxed())
}

fn create_validity(validity_array: Option<ArrayRef>, nullability: Nullability) -> Validity {
    match validity_array {
        Some(arr) => Validity::Array(arr),
        None => match nullability {
            Nullability::Nullable => Validity::AllValid,
            Nullability::NonNullable => Validity::NonNullable,
        },
    }
}

fn apply_validity(
    array: ArrayRef,
    validity_array: Option<ArrayRef>,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    if matches!(nullability, Nullability::Nullable) {
        let len = array.len();
        array.mask(create_validity(validity_array, nullability).to_array(len))
    } else {
        Ok(array)
    }
}

fn build_fixed_size_list(
    elements: ArrayRef,
    validity_array: Option<ArrayRef>,
    dtype: &DType,
    len: usize,
) -> VortexResult<ArrayRef> {
    let DType::FixedSizeList(_, list_size, nullability) = dtype else {
        return Err(vortex_err!(
            "FixedSizeListLayout requires FixedSizeList dtype, got {dtype}"
        ));
    };
    let validity = create_validity(validity_array, *nullability);
    Ok(FixedSizeListArray::try_new(elements, *list_size, validity, len)?.into_array())
}

fn predicate_array_to_mask(array: ArrayRef, session: &VortexSession) -> VortexResult<Mask> {
    let mut ctx = session.create_execution_ctx();
    array.null_as_false().execute(&mut ctx)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::FieldPath;
    use vortex_array::expr::gt;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::is_null;
    use vortex_array::expr::list_length;
    use vortex_array::expr::lit;
    use vortex_array::validity::Validity;

    use super::*;
    use crate::LayoutStrategy;
    use crate::layouts::fixed_size_list::writer::FixedSizeListLayoutStrategy;
    use crate::scan::split_by::SplitBy;
    use crate::segments::SegmentSource;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::SESSION;

    async fn write_layout(
        array: ArrayRef,
    ) -> VortexResult<(Arc<dyn SegmentSource>, crate::LayoutRef)> {
        let segments = Arc::new(TestSegments::default());
        let segments_ref: Arc<dyn SegmentSource> = Arc::<TestSegments>::clone(&segments);
        let (ptr, eof) = SequenceId::root().split();
        let stream = array.to_array_stream().sequenced(ptr);
        let layout = FixedSizeListLayoutStrategy::default()
            .write_stream(ArrayContext::empty(), segments, stream, eof, &SESSION)
            .await?;
        Ok((segments_ref, layout))
    }

    fn create_fsl(nullable: bool) -> ArrayRef {
        let validity = if nullable {
            Validity::Array(BoolArray::from_iter([true, false, true, true]).into_array())
        } else {
            Validity::NonNullable
        };
        FixedSizeListArray::new(
            PrimitiveArray::from_iter(0i32..8).into_array(),
            2,
            validity,
            4,
        )
        .into_array()
    }

    #[rstest]
    #[case::non_nullable(false)]
    #[case::nullable(true)]
    #[tokio::test]
    async fn projection_full_range(#[case] nullable: bool) -> VortexResult<()> {
        let fsl = create_fsl(nullable);
        let ctx = LayoutReaderContext::new();
        let (segments, layout) = write_layout(fsl.clone()).await?;
        let reader = layout.new_reader("".into(), segments, &SESSION, &ctx)?;

        let result = reader
            .projection_evaluation(&(0..4), &root(), MaskFuture::new_true(4))?
            .await?;

        let mut exec_ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(result, fsl, &mut exec_ctx);
        Ok(())
    }

    #[tokio::test]
    async fn projection_partial_range() -> VortexResult<()> {
        let fsl = create_fsl(true);
        let ctx = LayoutReaderContext::new();
        let (segments, layout) = write_layout(fsl.clone()).await?;
        let reader = layout.new_reader("".into(), segments, &SESSION, &ctx)?;

        let result = reader
            .projection_evaluation(&(1..4), &root(), MaskFuture::new_true(3))?
            .await?;
        let expected = fsl.slice(1..4)?;

        let mut exec_ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    #[tokio::test]
    async fn projection_sparse_mask() -> VortexResult<()> {
        let fsl = create_fsl(true);
        let ctx = LayoutReaderContext::new();
        let (segments, layout) = write_layout(fsl.clone()).await?;
        let reader = layout.new_reader("".into(), segments, &SESSION, &ctx)?;
        let mask = Mask::from_iter([true, false, false, true]);

        let result = reader
            .projection_evaluation(&(0..4), &root(), MaskFuture::ready(mask.clone()))?
            .await?;
        let expected = fsl.filter(mask)?;

        let mut exec_ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    #[tokio::test]
    async fn projection_degenerate_list_size_zero() -> VortexResult<()> {
        let fsl = FixedSizeListArray::new(
            PrimitiveArray::empty::<i32>(Nullability::NonNullable).into_array(),
            0,
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
            3,
        )
        .into_array();
        let ctx = LayoutReaderContext::new();
        let (segments, layout) = write_layout(fsl.clone()).await?;
        let reader = layout.new_reader("".into(), segments, &SESSION, &ctx)?;
        let mask = Mask::from_iter([false, true, true]);

        let result = reader
            .projection_evaluation(&(0..3), &root(), MaskFuture::ready(mask.clone()))?
            .await?;
        let expected = fsl.filter(mask)?;

        let mut exec_ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    #[tokio::test]
    async fn layout_splits_are_list_row_coordinates() -> VortexResult<()> {
        let fsl = create_fsl(true);
        let ctx = LayoutReaderContext::new();
        let (segments, layout) = write_layout(fsl).await?;
        let reader = layout.new_reader("".into(), segments, &SESSION, &ctx)?;

        let splits = SplitBy::Layout.splits(
            reader.as_ref(),
            &(0..4),
            &[FieldMask::Exact(FieldPath::root())],
        )?;

        assert_eq!(splits, vec![0, 4]);
        Ok(())
    }

    #[rstest]
    #[case::nullable(true, vec![true, false, true, true])]
    #[case::non_nullable(false, vec![true, true, true, true])]
    #[tokio::test]
    async fn projection_validity_class(
        #[case] nullable: bool,
        #[case] valid: Vec<bool>,
    ) -> VortexResult<()> {
        let fsl = create_fsl(nullable);
        let ctx = LayoutReaderContext::new();
        let (segments, layout) = write_layout(fsl).await?;
        let reader = layout.new_reader("".into(), segments, &SESSION, &ctx)?;

        let not_null = reader
            .projection_evaluation(&(0..4), &is_not_null(root()), MaskFuture::new_true(4))?
            .await?;
        let mut exec_ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(not_null, BoolArray::from_iter(valid.clone()), &mut exec_ctx);

        let null = reader
            .projection_evaluation(&(0..4), &is_null(root()), MaskFuture::new_true(4))?
            .await?;
        assert_arrays_eq!(
            null,
            BoolArray::from_iter(valid.iter().map(|v| !v).collect::<Vec<_>>()),
            &mut exec_ctx
        );
        Ok(())
    }

    #[tokio::test]
    async fn projection_list_length_preserves_validity() -> VortexResult<()> {
        let fsl = create_fsl(true);
        let ctx = LayoutReaderContext::new();
        let (segments, layout) = write_layout(fsl).await?;
        let reader = layout.new_reader("".into(), segments, &SESSION, &ctx)?;

        let result = reader
            .projection_evaluation(&(0..4), &list_length(root()), MaskFuture::new_true(4))?
            .await?;

        let expected =
            PrimitiveArray::from_option_iter::<u64, _>([Some(2), None, Some(2), Some(2)])
                .into_array();
        let mut exec_ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    #[tokio::test]
    async fn projection_list_length_applies_sparse_mask() -> VortexResult<()> {
        let fsl = create_fsl(true);
        let ctx = LayoutReaderContext::new();
        let (segments, layout) = write_layout(fsl).await?;
        let reader = layout.new_reader("".into(), segments, &SESSION, &ctx)?;
        let mask = Mask::from_iter([false, true, false, true]);

        let result = reader
            .projection_evaluation(&(0..4), &list_length(root()), MaskFuture::ready(mask))?
            .await?;

        let expected = PrimitiveArray::from_option_iter::<u64, _>([None, Some(2)]).into_array();
        let mut exec_ctx = SESSION.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    #[tokio::test]
    async fn filter_evaluation_list_length() -> VortexResult<()> {
        let fsl = create_fsl(true);
        let ctx = LayoutReaderContext::new();
        let (segments, layout) = write_layout(fsl).await?;
        let reader = layout.new_reader("".into(), segments, &SESSION, &ctx)?;

        let result = reader
            .filter_evaluation(
                &(0..4),
                &gt(list_length(root()), lit(1u64)),
                MaskFuture::new_true(4),
            )?
            .await?;

        assert_eq!(result, Mask::from_iter([true, false, true, true]));
        Ok(())
    }
}
