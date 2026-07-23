#![allow(clippy::expect_used)]
#![allow(clippy::tests_outside_test_module)]
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end correctness evidence for the zoned Bloom skipping index.

use std::num::NonZeroU8;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

use futures::FutureExt;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::field_path;
use vortex_array::stream::ArrayStreamExt;
use vortex_error::VortexResult;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_io::session::RuntimeSession;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::zoned::skip_index::SkipIndex;
use vortex_layout::layouts::zoned::skip_index::bloom::BloomOptions;
use vortex_layout::layouts::zoned::skip_index::bloom::BloomSkipIndex;
use vortex_layout::layouts::zoned::writer::ZonedLayoutOptions;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;
use vortex_layout::session::LayoutSession;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

const ZONE_LEN: usize = 256;
const NZONES: usize = 4;
const HIT: i64 = 502;
const MISS: i64 = 503;

fn bloom() -> BloomSkipIndex {
    BloomSkipIndex::new(BloomOptions::new(
        NonZeroUsize::new(1024).expect("1024 is non-zero"),
        NonZeroU8::new(5).expect("5 is non-zero"),
    ))
}

fn session(index: Option<&dyn SkipIndex>) -> VortexSession {
    let session = vortex_array::array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_file::register_default_encodings(&session);
    if let Some(index) = index {
        index.register(&session);
    }
    session
}

fn data() -> ArrayRef {
    data_with_shape(ZONE_LEN, NZONES, Some(MISS))
}

fn data_with_shape(zone_len: usize, nzones: usize, missing: Option<i64>) -> ArrayRef {
    let chunks = (0..nzones)
        .map(|zone| {
            let mut values = (0..zone_len)
                .map(|row| i64::try_from(row * nzones + zone).expect("test value fits i64"))
                .collect::<Vec<_>>();
            if let Some(missing) = missing
                && usize::try_from(missing).expect("missing value is non-negative") % nzones == zone
            {
                // Leave a hole inside every zone's min/max range so a MISS cannot be pruned by the
                // ordinary range stats. The bloom must provide the proof.
                values[usize::try_from(missing).expect("missing value is non-negative") / nzones] =
                    i64::try_from(zone_len * nzones + zone).expect("replacement fits i64");
            }
            StructArray::from_fields(&[("id", PrimitiveArray::from_iter(values).into_array())])
                .expect("valid test struct")
                .into_array()
        })
        .collect::<Vec<_>>();
    ChunkedArray::try_new(
        chunks,
        DType::struct_(
            [("id", DType::Primitive(PType::I64, Nullability::NonNullable))],
            Nullability::NonNullable,
        ),
    )
    .expect("valid chunked test data")
    .into_array()
}

fn low_card_data(zone_len: usize, nzones: usize, cardinality: i64) -> ArrayRef {
    let chunks = (0..nzones)
        .map(|_| {
            StructArray::from_fields(&[(
                "id",
                PrimitiveArray::from_iter((0..zone_len).map(|row| row as i64 % cardinality))
                    .into_array(),
            )])
            .expect("valid low-cardinality struct")
            .into_array()
        })
        .collect::<Vec<_>>();
    ChunkedArray::try_new(
        chunks,
        DType::struct_(
            [("id", DType::Primitive(PType::I64, Nullability::NonNullable))],
            Nullability::NonNullable,
        ),
    )
    .expect("valid low-cardinality chunked data")
    .into_array()
}

fn filter(value: i64) -> Expression {
    eq(get_item("id", root()), lit(value))
}

fn strategy(
    session: &VortexSession,
    index: Option<&dyn SkipIndex>,
    zone_len: usize,
) -> VortexResult<Arc<dyn LayoutStrategy>> {
    let mut options = ZonedLayoutOptions {
        block_size: NonZeroUsize::new(zone_len).expect("zone length is non-zero"),
        ..Default::default()
    };
    if let Some(index) = index {
        options = options.with_skip_index(index, &PType::I64.into(), session)?;
    }
    Ok(WriteStrategyBuilder::default()
        .with_field_zoned_options(field_path!(id), options)
        .build())
}

async fn scan(file: &vortex_file::VortexFile, value: i64) -> VortexResult<ArrayRef> {
    file.scan()?
        .with_filter(filter(value))
        .into_array_stream()?
        .read_all()
        .await
}

async fn write_file(
    session: &VortexSession,
    input: &ArrayRef,
    index: Option<&dyn SkipIndex>,
    zone_len: usize,
) -> VortexResult<Vec<u8>> {
    let mut bytes = Vec::new();
    session
        .write_options()
        .with_strategy(strategy(session, index, zone_len)?)
        .write(&mut bytes, input.to_array_stream())
        .await?;
    Ok(bytes)
}

#[tokio::test]
async fn bloom_roundtrip_prunes_and_unknown_reader_matches_full_scan() -> VortexResult<()> {
    let index = bloom();
    let write_session = session(Some(&index));
    let input = data();
    let bytes = write_file(&write_session, &input, Some(&index), ZONE_LEN).await?;

    // Fresh registered reader: prove the exact zones kept by both a hit and a miss.
    let read_session = session(Some(&index));
    let file = read_session.open_options().open_buffer(bytes.clone())?;
    let reader = file.layout_reader()?;
    let row_count = file.row_count();

    let hit_mask = reader
        .pruning_evaluation(
            &(0..row_count),
            &filter(HIT),
            Mask::new_true(usize::try_from(row_count)?),
        )?
        .await?;
    assert_eq!(hit_mask.true_count(), ZONE_LEN);
    assert!(hit_mask.iter().take(2 * ZONE_LEN).all(|keep| !keep));
    assert!(
        hit_mask
            .iter()
            .skip(2 * ZONE_LEN)
            .take(ZONE_LEN)
            .all(|keep| keep)
    );
    assert!(hit_mask.iter().skip(3 * ZONE_LEN).all(|keep| !keep));

    let miss_mask = reader
        .pruning_evaluation(
            &(0..row_count),
            &filter(MISS),
            Mask::new_true(usize::try_from(row_count)?),
        )?
        .await?;
    assert!(
        miss_mask.all_false(),
        "an absent value should prune every zone"
    );

    // Fresh unregistered reader: #8904's allow-unknown path bypasses the zone map and therefore
    // supplies the full-scan reference result instead of making the custom index a hard dependency.
    let full_scan_session = session(None);
    full_scan_session.allow_unknown();
    let full_scan_file = full_scan_session.open_options().open_buffer(bytes)?;

    let indexed_hit = scan(&file, HIT).await?;
    let full_scan_hit = scan(&full_scan_file, HIT).await?;
    assert_arrays_eq!(
        indexed_hit,
        full_scan_hit,
        &mut read_session.create_execution_ctx()
    );
    let expected_hit =
        StructArray::from_fields(&[("id", PrimitiveArray::from_iter([HIT]).into_array())])?
            .into_array();
    assert_arrays_eq!(
        full_scan_hit,
        expected_hit,
        &mut read_session.create_execution_ctx()
    );

    let indexed_miss = scan(&file, MISS).await?;
    let full_scan_miss = scan(&full_scan_file, MISS).await?;
    assert_arrays_eq!(
        indexed_miss,
        full_scan_miss,
        &mut read_session.create_execution_ctx()
    );
    assert_eq!(full_scan_miss.len(), 0);
    Ok(())
}

#[derive(Default)]
struct ReadCounts {
    requests: AtomicU64,
    bytes: AtomicU64,
    segment_ids: Mutex<HashSet<u32>>,
}

struct CountingSegmentSource {
    inner: Arc<dyn SegmentSource>,
    counts: Arc<ReadCounts>,
}

impl CountingSegmentSource {
    fn new(inner: Arc<dyn SegmentSource>) -> (Self, Arc<ReadCounts>) {
        let counts = Arc::new(ReadCounts::default());
        (
            Self {
                inner,
                counts: Arc::clone(&counts),
            },
            counts,
        )
    }
}

impl SegmentSource for CountingSegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let future = self.inner.request(id);
        let counts = Arc::clone(&self.counts);
        async move {
            let buffer = future.await?;
            counts.requests.fetch_add(1, Ordering::Relaxed);
            counts
                .bytes
                .fetch_add(buffer.len() as u64, Ordering::Relaxed);
            counts.segment_ids.lock().insert(*id);
            Ok(buffer)
        }
        .boxed()
    }
}

struct BenchRun {
    elapsed: Duration,
    requests: u64,
    segments: usize,
    bytes: u64,
    rows: usize,
}

async fn benchmark_scan(
    bytes: &[u8],
    session: &VortexSession,
    query: i64,
) -> VortexResult<BenchRun> {
    let file = session.open_options().open_buffer(bytes.to_vec())?;
    let (source, counts) = CountingSegmentSource::new(file.segment_source());
    let file = file.with_segment_source(Arc::new(source));
    let start = Instant::now();
    let result = scan(&file, query).await?;
    let elapsed = start.elapsed();
    Ok(BenchRun {
        elapsed,
        requests: counts.requests.load(Ordering::Relaxed),
        segments: counts.segment_ids.lock().len(),
        bytes: counts.bytes.load(Ordering::Relaxed),
        rows: result.len(),
    })
}

fn median(runs: &mut [BenchRun]) -> &BenchRun {
    runs.sort_by_key(|run| run.elapsed);
    &runs[runs.len() / 2]
}

/// Focused Bloom benchmark. Run with:
///
/// `cargo test --release -p vortex-file --test bloom_skip_index bloom_point_lookup_benchmark
/// -- --ignored --nocapture`
#[tokio::test]
#[ignore = "release-only benchmark"]
async fn bloom_point_lookup_benchmark() -> VortexResult<()> {
    const BENCH_ZONE_LEN: usize = 8192;
    const BENCH_NZONES: usize = 256;
    const ITERATIONS: usize = 9;
    const WARMUPS: usize = 2;

    let query =
        i64::try_from((BENCH_ZONE_LEN / 2) * BENCH_NZONES + 17).expect("benchmark query fits i64");
    let input = data_with_shape(BENCH_ZONE_LEN, BENCH_NZONES, None);
    let index = BloomSkipIndex::default();
    let baseline_session = session(None);
    let indexed_session = session(Some(&index));

    let baseline_bytes = write_file(&baseline_session, &input, None, BENCH_ZONE_LEN).await?;
    let indexed_bytes = write_file(&indexed_session, &input, Some(&index), BENCH_ZONE_LEN).await?;

    // Pruning diagnostics are separate from timing so reading the zone map here cannot warm the
    // timed reader. The query lies inside every zone's min/max range by construction.
    let diagnostic_file = indexed_session
        .open_options()
        .open_buffer(indexed_bytes.clone())?;
    let diagnostic_mask = diagnostic_file
        .layout_reader()?
        .pruning_evaluation(
            &(0..diagnostic_file.row_count()),
            &filter(query),
            Mask::new_true(usize::try_from(diagnostic_file.row_count())?),
        )?
        .await?;
    let zones_kept = diagnostic_mask.true_count() / BENCH_ZONE_LEN;
    let zones_pruned = BENCH_NZONES - zones_kept;
    assert!(
        zones_pruned >= BENCH_NZONES - 16,
        "the selective bloom benchmark should prune almost every zone"
    );

    for _ in 0..WARMUPS {
        let baseline = benchmark_scan(&baseline_bytes, &baseline_session, query).await?;
        let indexed = benchmark_scan(&indexed_bytes, &indexed_session, query).await?;
        assert_eq!(baseline.rows, 1);
        assert_eq!(indexed.rows, 1);
    }

    let mut baseline_runs = Vec::with_capacity(ITERATIONS);
    let mut indexed_runs = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        baseline_runs.push(benchmark_scan(&baseline_bytes, &baseline_session, query).await?);
        indexed_runs.push(benchmark_scan(&indexed_bytes, &indexed_session, query).await?);
    }

    let baseline = median(&mut baseline_runs);
    let indexed = median(&mut indexed_runs);
    println!(
        "bloom-bench rows={} zones={} zone_len={} query={} zones_pruned={} zones_kept={}",
        input.len(),
        BENCH_NZONES,
        BENCH_ZONE_LEN,
        query,
        zones_pruned,
        zones_kept
    );
    println!(
        "bloom-bench baseline file_bytes={} median_ms={:.3} requests={} segments={} segment_bytes={}",
        baseline_bytes.len(),
        baseline.elapsed.as_secs_f64() * 1000.0,
        baseline.requests,
        baseline.segments,
        baseline.bytes
    );
    println!(
        "bloom-bench indexed file_bytes={} median_ms={:.3} requests={} segments={} segment_bytes={}",
        indexed_bytes.len(),
        indexed.elapsed.as_secs_f64() * 1000.0,
        indexed.requests,
        indexed.segments,
        indexed.bytes
    );
    println!(
        "bloom-bench speedup={:.3}x byte_reduction={:.3}x",
        baseline.elapsed.as_secs_f64() / indexed.elapsed.as_secs_f64(),
        baseline.bytes as f64 / indexed.bytes as f64
    );

    // Negative control: a low-cardinality equality present in every zone cannot skip any data.
    let low_card_input = low_card_data(BENCH_ZONE_LEN, BENCH_NZONES, 16);
    let low_card_query = 7;
    let low_card_baseline_bytes =
        write_file(&baseline_session, &low_card_input, None, BENCH_ZONE_LEN).await?;
    let low_card_indexed_bytes = write_file(
        &indexed_session,
        &low_card_input,
        Some(&index),
        BENCH_ZONE_LEN,
    )
    .await?;

    let low_card_diagnostic_file = indexed_session
        .open_options()
        .open_buffer(low_card_indexed_bytes.clone())?;
    let low_card_mask = low_card_diagnostic_file
        .layout_reader()?
        .pruning_evaluation(
            &(0..low_card_diagnostic_file.row_count()),
            &filter(low_card_query),
            Mask::new_true(usize::try_from(low_card_diagnostic_file.row_count())?),
        )?
        .await?;
    let low_card_zones_kept = low_card_mask.true_count() / BENCH_ZONE_LEN;
    assert_eq!(
        low_card_zones_kept, BENCH_NZONES,
        "a value present in every zone cannot be pruned"
    );

    for _ in 0..WARMUPS {
        let baseline =
            benchmark_scan(&low_card_baseline_bytes, &baseline_session, low_card_query).await?;
        let indexed =
            benchmark_scan(&low_card_indexed_bytes, &indexed_session, low_card_query).await?;
        assert_eq!(baseline.rows, low_card_input.len() / 16);
        assert_eq!(indexed.rows, baseline.rows);
    }

    let mut low_card_baseline_runs = Vec::with_capacity(ITERATIONS);
    let mut low_card_indexed_runs = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        low_card_baseline_runs.push(
            benchmark_scan(&low_card_baseline_bytes, &baseline_session, low_card_query).await?,
        );
        low_card_indexed_runs
            .push(benchmark_scan(&low_card_indexed_bytes, &indexed_session, low_card_query).await?);
    }
    let low_card_baseline = median(&mut low_card_baseline_runs);
    let low_card_indexed = median(&mut low_card_indexed_runs);
    println!(
        "bloom-bench-low-card rows={} zones={} query={} zones_pruned={} zones_kept={}",
        low_card_input.len(),
        BENCH_NZONES,
        low_card_query,
        BENCH_NZONES - low_card_zones_kept,
        low_card_zones_kept
    );
    println!(
        "bloom-bench-low-card baseline file_bytes={} median_ms={:.3} requests={} segments={} segment_bytes={}",
        low_card_baseline_bytes.len(),
        low_card_baseline.elapsed.as_secs_f64() * 1000.0,
        low_card_baseline.requests,
        low_card_baseline.segments,
        low_card_baseline.bytes
    );
    println!(
        "bloom-bench-low-card indexed file_bytes={} median_ms={:.3} requests={} segments={} segment_bytes={}",
        low_card_indexed_bytes.len(),
        low_card_indexed.elapsed.as_secs_f64() * 1000.0,
        low_card_indexed.requests,
        low_card_indexed.segments,
        low_card_indexed.bytes
    );
    println!(
        "bloom-bench-low-card speedup={:.3}x byte_reduction={:.3}x",
        low_card_baseline.elapsed.as_secs_f64() / low_card_indexed.elapsed.as_secs_f64(),
        low_card_baseline.bytes as f64 / low_card_indexed.bytes as f64
    );
    Ok(())
}
