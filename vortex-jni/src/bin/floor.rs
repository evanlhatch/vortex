// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native-Rust "floor" lane for the vortex-jni read benchmark.
//!
//! Reads the SAME `.vortex` file that `VortexJniReadBenchmark` reads, but entirely in Rust: it
//! mirrors the JNI read path (session -> file -> scan -> array stream -> `execute_arrow`) using the
//! very same [`CurrentThreadRuntime`] the shipping JNI lib uses, and stops *before* the Arrow C Data
//! Interface export / JNI call / JVM vector access. The delta between this floor and the JMH numbers
//! is therefore attributable to the boundary (FFI marshaling + JNI + JVM), not to the format.
//!
//! For each op it reports rows/s normalized by the file's total row count (the same normalization
//! JMH uses via `@OperationsPerInvocation(ROWS)`), so JMH ops/s and floor rows/s are directly
//! comparable:
//!   * `fullScan`         — project all columns, no filter.
//!   * `projection`       — native projection of `id,y`.
//!   * `selectiveFilter`  — native filter `cat = 'alpha'` (~1/16 selectivity), all columns.
//!
//! Usage: `floor <path-to.vortex> [warmup_iters] [measured_iters]`.

use std::env::consts::ARCH;
use std::error::Error;
use std::hint::black_box;
use std::sync::LazyLock;
use std::time::Duration;
use std::time::Instant;

use futures::StreamExt;
use futures::pin_mut;
use vortex::VortexSessionDefault;
use vortex::array::VortexSessionExecute;
use vortex::array::arrow::ArrowSessionExt;
use vortex::expr::Expression;
use vortex::expr::col;
use vortex::expr::eq;
use vortex::expr::lit;
use vortex::expr::root;
use vortex::expr::select;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;

/// Current-thread runtime, identical to the one the JNI lib drives its scans on.
static RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);

/// Outcome of running one op: rows produced by the last iteration and per-iteration wall times.
struct OpResult {
    output_rows: u64,
    per_iter: Vec<Duration>,
}

/// Scan `file` once with the given projection/filter, materializing every chunk to Arrow in-process
/// (`execute_arrow`) exactly as the JNI path does before it exports across the C Data Interface.
/// Returns the number of rows produced (post-filter).
fn scan_once(
    file: &VortexFile,
    session: &VortexSession,
    projection: Expression,
    filter: Option<Expression>,
) -> Result<u64, Box<dyn Error>> {
    let stream = file
        .scan()?
        .with_projection(projection)
        .with_some_filter(filter)
        .into_array_stream()?;
    let rows = RUNTIME.block_on(async move {
        pin_mut!(stream);
        let mut ctx = session.create_execution_ctx();
        let mut rows: u64 = 0;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let arrow = session.arrow().execute_arrow(chunk, None, &mut ctx)?;
            rows += arrow.len() as u64;
            black_box(&arrow);
        }
        Ok::<u64, Box<dyn Error>>(rows)
    })?;
    Ok(rows)
}

/// Warm up, then time `iters` full scans of one op.
fn bench_op(
    file: &VortexFile,
    session: &VortexSession,
    projection: Expression,
    filter: Option<Expression>,
    warmup: usize,
    iters: usize,
) -> Result<OpResult, Box<dyn Error>> {
    for _ in 0..warmup {
        scan_once(file, session, projection.clone(), filter.clone())?;
    }
    let mut per_iter = Vec::with_capacity(iters);
    let mut output_rows = 0;
    for _ in 0..iters {
        let start = Instant::now();
        output_rows = scan_once(file, session, projection.clone(), filter.clone())?;
        per_iter.push(start.elapsed());
    }
    Ok(OpResult {
        output_rows,
        per_iter,
    })
}

/// Median of a set of durations, in seconds.
fn median_secs(durations: &[Duration]) -> f64 {
    let mut secs: Vec<f64> = durations.iter().map(Duration::as_secs_f64).collect();
    secs.sort_by(f64::total_cmp);
    let mid = secs.len() / 2;
    if secs.len() % 2 == 1 {
        secs[mid]
    } else {
        (secs[mid - 1] + secs[mid]) / 2.0
    }
}

fn report(name: &str, input_rows: u64, res: &OpResult) {
    let median = median_secs(&res.per_iter);
    let best = res
        .per_iter
        .iter()
        .map(Duration::as_secs_f64)
        .fold(f64::INFINITY, f64::min);
    let worst = res
        .per_iter
        .iter()
        .map(Duration::as_secs_f64)
        .fold(0.0_f64, f64::max);
    let rps_median = input_rows as f64 / median;
    let rps_best = input_rows as f64 / best;
    let rps_worst = input_rows as f64 / worst;
    println!(
        "floor {name}: input_rows={input_rows} output_rows={out} iters={n} \
         median_rows_per_s={rps_median:.0} best_rows_per_s={rps_best:.0} worst_rows_per_s={rps_worst:.0} \
         median_ms={ms:.3}",
        out = res.output_rows,
        n = res.per_iter.len(),
        ms = median * 1000.0,
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: floor <path-to.vortex> [warmup_iters] [measured_iters]")?;
    let warmup: usize = match args.next() {
        Some(v) => v.parse()?,
        None => 3,
    };
    let iters: usize = match args.next() {
        Some(v) => v.parse()?,
        None => 7,
    };

    // Build the session exactly as the JNI lib does, so encodings match the writer's session.
    let session = VortexSession::default().with_handle(RUNTIME.handle());
    vortex_parquet_variant::initialize(&session);

    let file = RUNTIME.block_on(
        session
            .open_options()
            .with_layout_reader_cache()
            .open_path(&path),
    )?;
    let total = file.row_count();
    if total == 0 {
        return Err(format!("file {path} reports zero rows").into());
    }
    let expected_alpha = total / 16;

    println!("floor: file={path} total_rows={total} warmup={warmup} iters={iters}");
    println!("floor: target_arch={ARCH}");

    // fullScan: all columns, no filter.
    let full = bench_op(&file, &session, root(), None, warmup, iters)?;
    if full.output_rows != total {
        return Err(format!(
            "fullScan produced {} rows, expected {total}",
            full.output_rows
        )
        .into());
    }
    report("fullScan", total, &full);

    // projection: native projection of id,y.
    let proj = select(vec!["id", "y"], root());
    let projection = bench_op(&file, &session, proj, None, warmup, iters)?;
    if projection.output_rows != total {
        return Err(format!(
            "projection produced {} rows, expected {total}",
            projection.output_rows
        )
        .into());
    }
    report("projection", total, &projection);

    // selectiveFilter: native filter cat = 'alpha' (~1/16), all columns.
    let filter = eq(col("cat"), lit("alpha"));
    let selective = bench_op(&file, &session, root(), Some(filter), warmup, iters)?;
    if selective.output_rows != expected_alpha {
        return Err(format!(
            "selectiveFilter produced {} rows, expected {expected_alpha}",
            selective.output_rows
        )
        .into());
    }
    report("selectiveFilter", total, &selective);

    Ok(())
}
