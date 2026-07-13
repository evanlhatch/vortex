// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::future::try_join;
use futures::future::try_join_all;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::fixed_size_list::FixedSizeListDataParts;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::kanal_ext::KanalExt;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::IntoLayout;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::fixed_size_list::FixedSizeListLayout;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStream;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Item carried on each child sub-stream: a sequenced, materialized chunk.
type ChildChunk = VortexResult<(SequenceId, ArrayRef)>;

/// Strategy for writing fixed-size-list arrays, with a fallback for other dtypes.
///
/// For fixed-size-list input the strategy transposes the whole column stream into `elements`
/// and optional `validity` sub-streams. Each input chunk is canonicalized to a
/// [`FixedSizeListArray`], then its children are streamed concurrently to independently
/// configurable child strategies.
#[derive(Clone)]
pub struct FixedSizeListLayoutStrategy {
    elements: Arc<dyn LayoutStrategy>,
    validity: Arc<dyn LayoutStrategy>,
    fallback: Arc<dyn LayoutStrategy>,
}

impl Default for FixedSizeListLayoutStrategy {
    fn default() -> Self {
        let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        Self {
            elements: Arc::clone(&flat),
            validity: Arc::clone(&flat),
            fallback: flat,
        }
    }
}

impl FixedSizeListLayoutStrategy {
    /// Strategy for the `elements` child.
    pub fn with_elements(mut self, elements: Arc<dyn LayoutStrategy>) -> Self {
        self.elements = elements;
        self
    }

    /// Strategy for the `validity` child, written only when the list dtype is nullable.
    pub fn with_validity(mut self, validity: Arc<dyn LayoutStrategy>) -> Self {
        self.validity = validity;
        self
    }

    /// Strategy for non-fixed-size-list input, which is forwarded unchanged.
    pub fn with_fallback(mut self, fallback: Arc<dyn LayoutStrategy>) -> Self {
        self.fallback = fallback;
        self
    }
}

#[async_trait]
impl LayoutStrategy for FixedSizeListLayoutStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        if !dtype.is_fixed_size_list() {
            return self
                .fallback
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }

        let is_nullable = dtype.is_nullable();
        let element_dtype = dtype
            .as_fixed_size_list_element_opt()
            .vortex_expect("DType is FixedSizeList")
            .as_ref()
            .clone();

        let (elements_tx, elements_rx) = kanal::bounded_async::<ChildChunk>(1);
        let (validity_tx, validity_rx) = if is_nullable {
            let (tx, rx) = kanal::bounded_async::<ChildChunk>(1);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };

        let fanout_fut =
            transpose_fixed_size_list_column(stream, session.clone(), elements_tx, validity_tx);

        let handle = session.handle();
        let mut child_specs: Vec<(
            DType,
            Arc<dyn LayoutStrategy>,
            kanal::AsyncReceiver<ChildChunk>,
        )> = vec![(element_dtype, Arc::clone(&self.elements), elements_rx)];
        if let Some(validity_rx) = validity_rx {
            child_specs.push((
                DType::Bool(Nullability::NonNullable),
                Arc::clone(&self.validity),
                validity_rx,
            ));
        }

        let layout_futures: Vec<_> = child_specs
            .into_iter()
            .map(|(child_dtype, strategy, rx)| {
                let child_stream =
                    SequentialStreamAdapter::new(child_dtype, rx.into_stream().boxed()).sendable();
                let child_eof = eof.split_off();
                let ctx = ctx.clone();
                let segment_sink = Arc::clone(&segment_sink);
                let session = session.clone();
                handle.spawn_nested(move |h| async move {
                    let session = session.with_handle(h);
                    strategy
                        .write_stream(ctx, segment_sink, child_stream, child_eof, &session)
                        .await
                })
            })
            .collect();

        let (row_count, layouts) = try_join(fanout_fut, try_join_all(layout_futures)).await?;
        let mut layouts = layouts.into_iter();
        let elements_layout = layouts.next().vortex_expect("elements layout present");
        let validity_layout =
            is_nullable.then(|| layouts.next().vortex_expect("validity layout present"));

        Ok(
            FixedSizeListLayout::new(row_count, dtype, elements_layout, validity_layout)
                .into_layout(),
        )
    }

    fn buffered_bytes(&self) -> u64 {
        let fsl_bytes = self.elements.buffered_bytes() + self.validity.buffered_bytes();
        fsl_bytes.max(self.fallback.buffered_bytes())
    }
}

/// Transpose a fixed-size-list column into `elements` and (when present) `validity` child
/// sub-streams. Errors surface to the caller, which joins this against the child writers, rather
/// than being hidden as an early channel close.
async fn transpose_fixed_size_list_column(
    mut stream: SendableSequentialStream,
    session: VortexSession,
    elements_tx: kanal::AsyncSender<ChildChunk>,
    validity_tx: Option<kanal::AsyncSender<ChildChunk>>,
) -> VortexResult<u64> {
    let mut exec_ctx = session.create_execution_ctx();
    let mut row_count = 0u64;
    let mut saw_chunk = false;

    while let Some(chunk) = stream.next().await {
        let (sequence_id, array) = chunk?;
        saw_chunk = true;
        let len = array.len();
        let FixedSizeListDataParts {
            elements, validity, ..
        } = canonicalize_to_fixed_size_list_parts(array, &mut exec_ctx)?;
        row_count += u64::try_from(len)?;

        let mut sp = sequence_id.descend();
        if elements_tx
            .send(Ok((sp.advance(), elements)))
            .await
            .is_err()
        {
            vortex_bail!("fixed-size-list elements writer finished before all chunks were sent");
        };

        if let Some(validity_tx) = &validity_tx {
            let validity = validity.execute_mask(len, &mut exec_ctx)?.into_array();
            if validity_tx
                .send(Ok((sp.advance(), validity)))
                .await
                .is_err()
            {
                vortex_bail!(
                    "fixed-size-list validity writer finished before all chunks were sent"
                );
            }
        }
    }

    if !saw_chunk {
        vortex_bail!("FixedSizeListLayoutStrategy needs at least one chunk");
    }

    Ok(row_count)
}

fn canonicalize_to_fixed_size_list_parts(
    array: ArrayRef,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<FixedSizeListDataParts> {
    Ok(array
        .execute::<FixedSizeListArray>(exec_ctx)?
        .into_data_parts())
}

#[cfg(test)]
mod tests {
    use futures::stream;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use super::*;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::SESSION;

    async fn write<S: LayoutStrategy>(strategy: &S, array: ArrayRef) -> VortexResult<LayoutRef> {
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let stream = array.to_array_stream().sequenced(ptr);
        strategy
            .write_stream(ArrayContext::empty(), segments, stream, eof, &SESSION)
            .await
    }

    async fn write_chunks<S: LayoutStrategy>(
        strategy: &S,
        dtype: DType,
        chunks: Vec<ArrayRef>,
    ) -> VortexResult<LayoutRef> {
        let segments = Arc::new(TestSegments::default());
        let (mut ptr, eof) = SequenceId::root().split();
        let chunks = chunks
            .into_iter()
            .map(move |chunk| Ok((ptr.advance(), chunk)));
        let stream = SequentialStreamAdapter::new(dtype, stream::iter(chunks).boxed()).sendable();
        strategy
            .write_stream(ArrayContext::empty(), segments, stream, eof, &SESSION)
            .await
    }

    fn i32_fsl_dtype(nullable: bool) -> DType {
        DType::FixedSizeList(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            2,
            if nullable {
                Nullability::Nullable
            } else {
                Nullability::NonNullable
            },
        )
    }

    fn create_fsl(validity: Validity) -> ArrayRef {
        FixedSizeListArray::new(buffer![1i32, 2, 3, 4, 5, 6].into_array(), 2, validity, 3)
            .into_array()
    }

    #[tokio::test]
    async fn basic_non_nullable_input() -> VortexResult<()> {
        let layout = write(
            &FixedSizeListLayoutStrategy::default(),
            create_fsl(Validity::NonNullable),
        )
        .await?;

        assert_eq!(
            layout.display_tree().to_string(),
            "vortex.fixed_size_list, dtype: fixed_size_list(i32)[2], children: 1\n\
             └── elements: vortex.flat, dtype: i32, segment: 0\n"
        );
        Ok(())
    }

    #[tokio::test]
    async fn basic_nullable_input() -> VortexResult<()> {
        let layout = write(
            &FixedSizeListLayoutStrategy::default(),
            create_fsl(Validity::Array(
                BoolArray::from_iter([true, false, true]).into_array(),
            )),
        )
        .await?;

        assert_eq!(
            layout.display_tree().to_string(),
            "vortex.fixed_size_list, dtype: fixed_size_list(i32)[2]?, children: 2\n\
             ├── elements: vortex.flat, dtype: i32, segment: 0\n\
             └── validity: vortex.flat, dtype: bool, segment: 1\n"
        );
        Ok(())
    }

    #[tokio::test]
    async fn chunked_input_with_chunked_child_strategies_succeeds() -> VortexResult<()> {
        let chunk0 = FixedSizeListArray::new(
            buffer![1i32, 2, 3, 4].into_array(),
            2,
            Validity::Array(BoolArray::from_iter([true, false]).into_array()),
            2,
        )
        .into_array();
        let chunk1 = FixedSizeListArray::new(
            buffer![5i32, 6].into_array(),
            2,
            Validity::Array(BoolArray::from_iter([true]).into_array()),
            1,
        )
        .into_array();

        let child_strategy: Arc<dyn LayoutStrategy> =
            Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()));
        let strategy = FixedSizeListLayoutStrategy::default()
            .with_elements(Arc::clone(&child_strategy))
            .with_validity(child_strategy);

        let layout = write_chunks(&strategy, i32_fsl_dtype(true), vec![chunk0, chunk1]).await?;

        assert_eq!(layout.row_count(), 3);
        insta::assert_snapshot!(layout.display_tree(), @"
        vortex.fixed_size_list, dtype: fixed_size_list(i32)[2]?, children: 2
        ├── elements: vortex.chunked, dtype: i32, children: 2
        │   ├── [0]: vortex.flat, dtype: i32, segment: 0
        │   └── [1]: vortex.flat, dtype: i32, segment: 1
        └── validity: vortex.chunked, dtype: bool, children: 2
            ├── [0]: vortex.flat, dtype: bool, segment: 2
            └── [1]: vortex.flat, dtype: bool, segment: 3
        ");
        Ok(())
    }

    #[tokio::test]
    async fn non_fixed_size_list_input_routes_to_fallback() -> VortexResult<()> {
        let primitive = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        let layout = write(&FixedSizeListLayoutStrategy::default(), primitive).await?;
        assert_eq!(
            layout.display_tree().to_string(),
            "vortex.flat, dtype: i32, segment: 0\n"
        );
        Ok(())
    }

    #[tokio::test]
    async fn empty_stream_errors() {
        let segments = Arc::new(TestSegments::default());
        let (_, eof) = SequenceId::root().split();
        let empty = stream::empty::<VortexResult<(SequenceId, ArrayRef)>>().boxed();
        let stream = SequentialStreamAdapter::new(
            DType::FixedSizeList(
                Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
                2,
                Nullability::NonNullable,
            ),
            empty,
        )
        .sendable();

        let res = FixedSizeListLayoutStrategy::default()
            .write_stream(ArrayContext::empty(), segments, stream, eof, &SESSION)
            .await;
        assert!(res.is_err());
    }
}
