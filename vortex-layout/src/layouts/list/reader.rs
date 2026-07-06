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
use vortex_array::arrays::ListArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldMask;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::root;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::ArrayFuture;
use crate::LayoutReader;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::RowSplits;
use crate::SplitRange;
use crate::layouts::list::ListLayout;
use crate::layouts::list::expr::ListChildrenNeeded;
use crate::layouts::list::expr::get_necessary_list_children;
use crate::layouts::list::expr::rewrite_offsets_expr;
use crate::layouts::list::expr::rewrite_validity_expr;
use crate::segments::SegmentSource;

type OptionalArrayFuture = BoxFuture<'static, VortexResult<Option<ArrayRef>>>;

/// The threshold of mask density below which we push the input mask into projection evaluation,
/// and above which we evaluate the expression over all rows and intersect afterward.
const EXPR_EVAL_THRESHOLD: f64 = 0.2;

/// Reader for [`ListLayout`].
#[derive(Clone)]
pub struct ListReader {
    layout: ListLayout,
    name: Arc<str>,
    session: VortexSession,
    elements: LayoutReaderRef,
    offsets: LayoutReaderRef,
    validity: Option<LayoutReaderRef>,
}

impl ListReader {
    pub(super) fn try_new(
        layout: ListLayout,
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
        let offsets = layout.offsets().new_reader(
            format!("{name}.offsets").into(),
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
            offsets,
            validity,
        })
    }

    /// Projection for [`ListChildrenNeeded::Validity`] expressions (`is_null` / `is_not_null` of
    /// the list): reads only the validity child, synthesizing all-valid for a non-nullable list,
    /// and never touches the offsets or elements.
    fn project_validity(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let validity_reader = self.validity.clone();
        let nullability = self.layout.dtype().nullability();
        let row_range = row_range.clone();
        // Evaluate the rewritten expression against the validity bool array (true == valid row).
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

            let validity = create_validity(validity_array, nullability).to_array(out_len);
            validity.apply(&rewritten)
        }
        .boxed())
    }

    /// Projection for [`ListChildrenNeeded::All`] expressions: materializes the list and applies
    /// the expression.
    ///
    /// This is a single **whole-chunk** read: the entire `elements` buffer, `offsets`, and
    /// `validity` for the chunk are fetched concurrently — there is no offsets→elements round-trip
    /// because the elements bound is the whole buffer (`offsets[0] == 0` within a chunk). The
    /// assembled list is sliced to `row_range` and filtered by the caller mask in memory. Reading
    /// the whole chunk keeps this one simple, allocation-light path with trivial nested
    /// composition; the chunk is the unit of random-access granularity (controlled by the writer's
    /// chunk size).
    fn project_elements(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let chunk_row_count = self.layout.row_count();
        let elements_row_count = self.elements.row_count();
        let nullability = self.layout.dtype().nullability();
        let expr = expr.clone();
        let row_range = row_range.clone();

        // Fire all three child reads up front so they run concurrently and overlap the mask await.
        // Offsets has one extra entry (`n + 1`).
        let offsets_fut = self.offsets.projection_evaluation(
            &(0..chunk_row_count + 1),
            &root(),
            MaskFuture::new_true(usize::try_from(chunk_row_count + 1)?),
        )?;
        let elements_fut = self.elements.projection_evaluation(
            &(0..elements_row_count),
            &root(),
            MaskFuture::new_true(usize::try_from(elements_row_count)?),
        )?;
        let validity_fut = fetch_validity(
            self.validity.as_ref(),
            &(0..chunk_row_count),
            MaskFuture::new_true(usize::try_from(chunk_row_count)?),
        )?;

        Ok(async move {
            let (offsets, elements, validity) = try_join!(offsets_fut, elements_fut, validity_fut)?;
            let list =
                ListArray::try_new(elements, offsets, create_validity(validity, nullability))?
                    .into_array();

            // Slice the whole-chunk list down to the requested row range.
            let list = if row_range.start == 0 && row_range.end == chunk_row_count {
                list
            } else {
                list.slice(usize::try_from(row_range.start)?..usize::try_from(row_range.end)?)?
            };

            // Filter before applying the expression: the expression may depend on the filtered
            // rows being removed (e.g. `cast(a, u8) where a < 256`).
            let mask = mask.await?;
            let list = if mask.all_true() {
                list
            } else {
                list.filter(mask)?
            };
            list.apply(&expr)
        }
        .boxed())
    }

    /// Projection for [`ListChildrenNeeded::OffsetsAndValidity`] expressions (`list_length(root())`
    /// and expressions composed from it): reads offsets and list validity, but never touches
    /// element values.
    fn project_offsets(
        &self,
        row_range: &Range<u64>,
        expr: &Expression,
        mask: MaskFuture,
    ) -> VortexResult<ArrayFuture> {
        let offsets = self.fetch_offsets(row_range)?;
        let reader = self.clone();
        let row_range = row_range.clone();
        let rewritten = rewrite_offsets_expr(expr)?;

        Ok(async move {
            let mask = mask.await?;
            let row_count = usize::try_from(row_range.end - row_range.start)?;
            let nullability = reader.layout.dtype().nullability();

            let validity_mask = if mask.all_true() {
                MaskFuture::new_true(row_count)
            } else {
                MaskFuture::ready(mask.clone())
            };
            let validity_fut = fetch_validity(reader.validity.as_ref(), &row_range, validity_mask)?;

            let offsets = offsets.await?;
            let lengths = list_lengths_from_offsets(offsets)?;
            let lengths = if mask.all_true() {
                lengths
            } else {
                lengths.filter(mask)?
            };
            let validity = validity_fut.await?;
            let lengths = apply_lengths_validity(lengths, validity, nullability)?;

            lengths.apply(&rewritten)
        }
        .boxed())
    }

    /// Fire the offsets read for `row_range`. The offsets child has an extra entry, so reading
    /// `row_range` maps to offsets in `[row_range.start..row_range.end + 1)`.
    fn fetch_offsets(&self, row_range: &Range<u64>) -> VortexResult<ArrayFuture> {
        let offsets_range = row_range.start..(row_range.end + 1);
        let offsets_count = usize::try_from(offsets_range.end - offsets_range.start)?;
        self.offsets.projection_evaluation(
            &offsets_range,
            &root(),
            MaskFuture::new_true(offsets_count),
        )
    }
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

impl LayoutReader for ListReader {
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
        self.offsets
            .register_splits(field_mask, split_range, splits)?;
        if let Some(validity) = &self.validity {
            validity.register_splits(field_mask, split_range, splits)?;
        }
        Ok(())
    }

    fn pruning_evaluation(
        &self,
        _row_range: &Range<u64>,
        _expr: &Expression,
        mask: Mask,
    ) -> VortexResult<MaskFuture> {
        // All stats-based pruning should already be done upstream.
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
        // Read as little as possible based on which list children the expression needs.
        match get_necessary_list_children(expr) {
            ListChildrenNeeded::Validity => self.project_validity(row_range, expr, mask),
            ListChildrenNeeded::OffsetsAndValidity => self.project_offsets(row_range, expr, mask),
            ListChildrenNeeded::All => self.project_elements(row_range, expr, mask),
        }
    }
}

/// Fetch the validity child for `row_range` under `mask`, yielding `None` for a non-nullable list
/// (which has no validity child).
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

/// Compute `offsets[i + 1] - offsets[i]` as the unmasked list length values.
fn list_lengths_from_offsets(offsets: ArrayRef) -> VortexResult<ArrayRef> {
    let len = offsets.len().saturating_sub(1);
    offsets
        .slice(1..offsets.len())?
        .binary(offsets.slice(0..len)?, Operator::Sub)
}

fn apply_lengths_validity(
    lengths: ArrayRef,
    validity: Option<ArrayRef>,
    nullability: Nullability,
) -> VortexResult<ArrayRef> {
    let len = lengths.len();
    let lengths = lengths.cast(DType::Primitive(PType::U64, nullability))?;

    if matches!(nullability, Nullability::Nullable) {
        lengths.mask(create_validity(validity, nullability).to_array(len))
    } else {
        Ok(lengths)
    }
}

fn predicate_array_to_mask(array: ArrayRef, session: &VortexSession) -> VortexResult<Mask> {
    let mut ctx = session.create_execution_ctx();
    array.null_as_false().execute(&mut ctx)
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::expr::cast;
    use vortex_array::expr::gt;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::is_null;
    use vortex_array::expr::list_length;
    use vortex_array::expr::lit;
    use vortex_buffer::buffer;
    use vortex_io::session::RuntimeSession;
    use vortex_io::session::RuntimeSessionExt;

    use super::*;
    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::layouts::list::writer::ListLayoutStrategy;
    use crate::segments::SegmentSource;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::session::LayoutSession;
    use crate::test::SESSION;

    /// Validity-class projections (`is_null` / `is_not_null` of the list) round-trip through the
    /// validity-only read path, for both nullable and non-nullable lists.
    #[rstest]
    // `create_basic_list_array(true)` has validity `[true, false, true]`.
    #[case::nullable(true, vec![true, false, true])]
    #[case::non_nullable(false, vec![true, true, true])]
    #[tokio::test]
    async fn projection_validity_class(
        #[case] nullable: bool,
        #[case] valid: Vec<bool>,
    ) -> VortexResult<()> {
        let list = create_basic_list_array(nullable);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let not_null = reader
            .projection_evaluation(&(0..3), &is_not_null(root()), MaskFuture::new_true(3))?
            .await?;
        let mut exec_ctx = session.create_execution_ctx();
        assert_arrays_eq!(not_null, BoolArray::from_iter(valid.clone()), &mut exec_ctx);

        let is_null_res = reader
            .projection_evaluation(&(0..3), &is_null(root()), MaskFuture::new_true(3))?
            .await?;
        assert_arrays_eq!(
            is_null_res,
            BoolArray::from_iter(valid.iter().map(|v| !v).collect::<Vec<_>>()),
            &mut exec_ctx
        );

        Ok(())
    }

    #[tokio::test]
    async fn projection_list_length_reads_offsets() -> VortexResult<()> {
        let list = create_basic_list_array(false);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let result = reader
            .projection_evaluation(&(0..3), &list_length(root()), MaskFuture::new_true(3))?
            .await?;

        let mut exec_ctx = session.create_execution_ctx();
        assert_arrays_eq!(result, buffer![2u64, 2, 1].into_array(), &mut exec_ctx);
        Ok(())
    }

    #[tokio::test]
    async fn projection_list_length_preserves_validity() -> VortexResult<()> {
        let list = create_basic_list_array(true);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let result = reader
            .projection_evaluation(&(0..3), &list_length(root()), MaskFuture::new_true(3))?
            .await?;

        let expected =
            PrimitiveArray::from_option_iter::<u64, _>([Some(2), None, Some(1)]).into_array();
        let mut exec_ctx = session.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    #[tokio::test]
    async fn projection_list_length_applies_sparse_mask() -> VortexResult<()> {
        let list = create_basic_list_array(true);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let mask = Mask::from_iter([false, true, true]);
        let result = reader
            .projection_evaluation(&(0..3), &list_length(root()), MaskFuture::ready(mask))?
            .await?;

        let expected = PrimitiveArray::from_option_iter::<u64, _>([None, Some(1)]).into_array();
        let mut exec_ctx = session.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    #[tokio::test]
    async fn projection_cast_list_length() -> VortexResult<()> {
        let list = create_basic_list_array(true);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let expr = cast(
            list_length(root()),
            DType::Primitive(PType::I64, Nullability::Nullable),
        );
        let result = reader
            .projection_evaluation(&(0..3), &expr, MaskFuture::new_true(3))?
            .await?;

        let expected =
            PrimitiveArray::from_option_iter::<i64, _>([Some(2), None, Some(1)]).into_array();
        let mut exec_ctx = session.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    #[tokio::test]
    async fn filter_evaluation_list_length() -> VortexResult<()> {
        let list = create_basic_list_array(true);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let result = reader
            .filter_evaluation(
                &(0..3),
                &gt(list_length(root()), lit(1u64)),
                MaskFuture::new_true(3),
            )?
            .await?;

        assert_eq!(result, Mask::from_iter([true, false, false]));
        Ok(())
    }

    #[rstest]
    #[case::is_not_null_nullable(true, is_not_null(root()), Mask::from_iter([true, false, true]))]
    #[case::is_not_null_non_nullable(false, is_not_null(root()), Mask::new_true(3))]
    #[case::is_null_nullable(true, is_null(root()), Mask::from_iter([false, true, false]))]
    #[case::is_null_non_nullable(false, is_null(root()), Mask::new_false(3))]
    #[tokio::test]
    async fn filter_evaluation_validity_class(
        #[case] nullable: bool,
        #[case] expr: Expression,
        #[case] expected: Mask,
    ) -> VortexResult<()> {
        let list = create_basic_list_array(nullable);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let result = reader
            .filter_evaluation(&(0..3), &expr, MaskFuture::new_true(3))?
            .await?;

        assert_eq!(result, expected);
        Ok(())
    }

    #[tokio::test]
    async fn filter_evaluation_intersects_with_input_mask() -> VortexResult<()> {
        let list = create_basic_list_array(true);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let input_mask = Mask::from_iter([true, true, false]);
        let result = reader
            .filter_evaluation(&(0..3), &is_not_null(root()), MaskFuture::ready(input_mask))?
            .await?;

        assert_eq!(result, Mask::from_iter([true, false, false]));
        Ok(())
    }

    #[tokio::test]
    async fn filter_evaluation_sparse_mask_maps_by_rank() -> VortexResult<()> {
        let list = create_six_list_array();
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let input_mask = Mask::from_iter([false, false, false, false, true, false]);
        let result = reader
            .filter_evaluation(&(0..6), &is_not_null(root()), MaskFuture::ready(input_mask))?
            .await?;

        assert_eq!(
            result,
            Mask::from_iter([false, false, false, false, true, false])
        );
        Ok(())
    }

    fn flat_list_strategy() -> ListLayoutStrategy {
        ListLayoutStrategy::default()
    }

    fn layout_test_session() -> VortexSession {
        vortex_array::array_session()
            .with::<LayoutSession>()
            .with::<RuntimeSession>()
    }

    async fn write_layout<S: LayoutStrategy>(
        strategy: &S,
        array: ArrayRef,
    ) -> VortexResult<(Arc<dyn SegmentSource>, LayoutRef, VortexSession)> {
        let session = layout_test_session().with_tokio();
        let segments = Arc::new(TestSegments::default());
        let segments_ref: Arc<dyn SegmentSource> = Arc::<TestSegments>::clone(&segments);
        let (ptr, eof) = SequenceId::root().split();
        let stream = array.to_array_stream().sequenced(ptr);
        let layout = strategy
            .write_stream(ArrayContext::empty(), segments, stream, eof, &session)
            .await?;
        Ok((segments_ref, layout, session))
    }

    fn materialize_u32_array(array: ArrayRef) -> Vec<u32> {
        let mut ctx = SESSION.create_execution_ctx();
        array
            .execute::<PrimitiveArray>(&mut ctx)
            .unwrap()
            .as_slice::<u32>()
            .to_vec()
    }

    fn create_basic_list_array(nullable: bool) -> ArrayRef {
        let validity = if nullable {
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array())
        } else {
            Validity::NonNullable
        };

        ListArray::try_new(
            buffer![1i32, 2, 3, 4, 5].into_array(),
            buffer![0u32, 2, 4, 5].into_array(),
            validity,
        )
        .expect("array is valid")
        .into_array()
    }

    fn create_six_list_array() -> ArrayRef {
        let validity = Validity::Array(
            BoolArray::from_iter([true, false, true, false, true, true]).into_array(),
        );

        ListArray::try_new(
            buffer![1i32, 2, 3, 4, 5, 6].into_array(),
            buffer![0u32, 1, 2, 3, 4, 5, 6].into_array(),
            validity,
        )
        .expect("array is valid")
        .into_array()
    }

    #[tokio::test]
    async fn fetch_offsets_includes_extra_endpoint() -> VortexResult<()> {
        let list = create_basic_list_array(false);

        let (segments, layout, session) = write_layout(&flat_list_strategy(), list).await?;
        let ctx = LayoutReaderContext::new();
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;
        let reader = reader
            .as_any()
            .downcast_ref::<ListReader>()
            .expect("ListReader");

        let offsets = reader.fetch_offsets(&(1..3))?.await?;
        assert_eq!(materialize_u32_array(offsets), vec![2, 4, 5]);

        Ok(())
    }

    #[rstest]
    #[case::full_range(0..3, false)]
    #[case::partial_start(0..2, false)]
    #[case::partial_end(1..3, false)]
    #[case::middle_single(1..2, false)]
    #[case::empty_range(1..1, false)]
    #[case::full_range_null(0..3, true)]
    #[tokio::test]
    async fn projection_evaluation_round_trips(
        #[case] row_range: Range<u64>,
        #[case] nullable: bool,
    ) -> VortexResult<()> {
        let list = create_basic_list_array(nullable);
        let ctx = LayoutReaderContext::new();

        let len = usize::try_from(row_range.end - row_range.start)?;
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list.clone()).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let result = reader
            .projection_evaluation(&row_range, &root(), MaskFuture::new_true(len))?
            .await?;

        let expected =
            list.slice(usize::try_from(row_range.start)?..usize::try_from(row_range.end)?)?;
        let mut exec_ctx = session.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    #[tokio::test]
    async fn projection_evaluation_applies_mask() -> VortexResult<()> {
        let list = create_basic_list_array(false);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list.clone()).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let mask = Mask::from_iter([true, false, true]);
        let result = reader
            .projection_evaluation(&(0..3), &root(), MaskFuture::ready(mask.clone()))?
            .await?;

        let expected = list.filter(mask)?;
        let mut exec_ctx = session.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    /// Build a list with 5 rows and lengths [2, 3, 0, 3, 2].
    fn create_wider_list_array(nullable: bool) -> ArrayRef {
        let validity = if nullable {
            Validity::Array(BoolArray::from_iter([true, true, false, true, true]).into_array())
        } else {
            Validity::NonNullable
        };
        ListArray::try_new(
            buffer![10i32, 11, 20, 21, 22, 30, 31, 32, 40, 41].into_array(),
            buffer![0u32, 2, 5, 5, 8, 10].into_array(),
            validity,
        )
        .expect("array is valid")
        .into_array()
    }

    /// Sparse row masks over a whole chunk round-trip: the whole-chunk read is filtered by the
    /// mask in memory.
    #[rstest]
    #[case::single_middle(Mask::from_iter([false, false, false, true, false]), false)]
    #[case::two_far_apart(Mask::from_iter([true, false, false, true, false]), false)]
    #[case::boundaries(Mask::from_iter([true, false, false, false, true]), false)]
    #[case::kept_empty_row(Mask::from_iter([false, false, true, false, false]), false)]
    #[case::sparse_nullable(Mask::from_iter([true, false, true, false, true]), true)]
    #[case::all_false(Mask::new_false(5), false)]
    #[tokio::test]
    async fn projection_evaluation_sparse_mask_round_trips(
        #[case] mask: Mask,
        #[case] nullable: bool,
    ) -> VortexResult<()> {
        let list = create_wider_list_array(nullable);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list.clone()).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let result = reader
            .projection_evaluation(&(0..5), &root(), MaskFuture::ready(mask.clone()))?
            .await?;

        let expected = list.filter(mask)?;
        let mut exec_ctx = session.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }

    /// A partial (sub-chunk) row range still returns exactly that range: the whole chunk is read,
    /// then sliced.
    #[tokio::test]
    async fn projection_evaluation_partial_range_slices_whole_chunk() -> VortexResult<()> {
        let list = create_wider_list_array(false);
        let ctx = LayoutReaderContext::new();
        let (segments, layout, session) = write_layout(&flat_list_strategy(), list.clone()).await?;
        let reader = layout.new_reader("".into(), segments, &session, &ctx)?;

        let result = reader
            .projection_evaluation(&(1..4), &root(), MaskFuture::new_true(3))?
            .await?;

        let expected = list.slice(1..4)?;
        let mut exec_ctx = session.create_execution_ctx();
        assert_arrays_eq!(result, expected, &mut exec_ctx);
        Ok(())
    }
}
