// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pure-Rust reproducer for the low-cardinality Utf8View equality-filter under-count.
//!
//! Builds a table shaped exactly like `VortexJniReadBenchmark.writeTable` (2M rows, `cat` cycling
//! 16 low-cardinality values with "alpha" one of them), writes it with the bench's DEFAULT write
//! config (`session.write_options()`), then reads it back three ways and prints:
//!   (a) fullScan -> materialize `cat` -> count == "alpha"
//!   (b) filter eq(col("cat"), lit("alpha")) -> count
//!   (c) total rows
//! Expected alpha = ROWS / 16.
//!
//! Knobs (env vars):
//!   UF_ROWS       total rows            (default 2_000_000)
//!   UF_CHUNK      write chunk rows      (default 65536)
//!   UF_VIEW       cat/tag as Utf8View   (default 1; 0 => Utf8)
//!   UF_NCATS      distinct cat values   (default 16)
//!   UF_KEEP       keep file at this path instead of a temp file (default: temp, deleted)

use std::env::consts::ARCH;
use std::error::Error;
use std::sync::LazyLock;

use arrow_array::Array as ArrowArray;
use arrow_array::Float64Array;
use arrow_array::Int64Array;
use arrow_array::RecordBatch;
use arrow_array::StringArray;
use arrow_array::StringViewArray;
use arrow_array::StructArray as ArrowStructArray;
use arrow_schema::DataType;
use arrow_schema::Field as ArrowField;
use arrow_schema::Schema as ArrowSchema;
use futures::StreamExt;
use futures::pin_mut;
use std::sync::Arc;
use vortex::VortexSessionDefault;
use vortex::array::VortexSessionExecute;
use vortex::array::arrow::ArrowSessionExt;
use vortex::error::VortexError;
use vortex::expr::col;
use vortex::expr::eq;
use vortex::expr::lit;
use vortex::expr::root;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::runtime::BlockingRuntime;
use vortex::io::runtime::current::CurrentThreadRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::array::iter::ArrayIteratorAdapter;
use vortex::layout::scan::split_by::SplitBy;
use vortex::session::VortexSession;

/// Read the optional `UF_SPLIT` override: `SplitBy::RowCount(n)` when set, else layout default.
fn split_override() -> Option<SplitBy> {
    std::env::var("UF_SPLIT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(SplitBy::RowCount)
}

/// Apply optional scan knobs from env: UF_SPLIT (RowCount), UF_CONC (concurrency), UF_ORDERED (0/1).
fn apply_scan_knobs(
    mut scan: vortex::layout::scan::scan_builder::ScanBuilder<vortex::array::ArrayRef>,
) -> vortex::layout::scan::scan_builder::ScanBuilder<vortex::array::ArrayRef> {
    if let Some(sb) = split_override() {
        scan = scan.with_split_by(sb);
    }
    if let Some(c) = std::env::var("UF_CONC").ok().and_then(|v| v.parse::<usize>().ok()) {
        scan = scan.with_concurrency(c);
    }
    if let Some(o) = std::env::var("UF_ORDERED").ok().and_then(|v| v.parse::<usize>().ok()) {
        scan = scan.with_ordered(o != 0);
    }
    scan
}

static RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);

const CATS: [&str; 16] = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
    "kilo", "lima", "mike", "november", "oscar", "papa",
];

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn string_field(name: &str, view: bool) -> ArrowField {
    let dt = if view {
        DataType::Utf8View
    } else {
        DataType::Utf8
    };
    ArrowField::new(name, dt, true)
}

fn string_col(values: Vec<Option<String>>, view: bool, block: usize) -> Arc<dyn ArrowArray> {
    if view {
        if block > 0 {
            // Build with a small block size so non-inline strings spill across MANY variadic data
            // buffers (views with buffer_index 0,1,2,...), mimicking Arrow-Java ViewVarCharVector.
            let mut b = arrow_array::builder::StringViewBuilder::new()
                .with_fixed_block_size(block as u32);
            for v in &values {
                b.append_option(v.as_deref());
            }
            Arc::new(b.finish()) as Arc<dyn ArrowArray>
        } else {
            Arc::new(StringViewArray::from_iter(values)) as Arc<dyn ArrowArray>
        }
    } else {
        Arc::new(StringArray::from_iter(values)) as Arc<dyn ArrowArray>
    }
}

/// Count the number of `cat` == "alpha" entries in an executed Arrow struct array.
fn count_alpha_in_struct(arrow: &Arc<dyn ArrowArray>) -> Result<usize, Box<dyn Error>> {
    let st = arrow
        .as_any()
        .downcast_ref::<ArrowStructArray>()
        .ok_or("execute_arrow output was not a struct array")?;
    let cat = st
        .column_by_name("cat")
        .ok_or("no 'cat' column in output")?;
    count_alpha(cat.as_ref())
}

/// Accumulate a distribution of `cat` values (and nulls) from an executed struct array.
fn tally_cat(
    arrow: &Arc<dyn ArrowArray>,
    hist: &mut std::collections::BTreeMap<String, u64>,
    nulls: &mut u64,
) -> Result<(), Box<dyn Error>> {
    let st = arrow
        .as_any()
        .downcast_ref::<ArrowStructArray>()
        .ok_or("not a struct array")?;
    let cat = st.column_by_name("cat").ok_or("no 'cat' column")?;
    let mut push = |v: Option<&str>| match v {
        Some(s) => *hist.entry(s.to_string()).or_insert(0) += 1,
        None => *nulls += 1,
    };
    if let Some(v) = cat.as_any().downcast_ref::<StringViewArray>() {
        for s in v.iter() {
            push(s);
        }
    } else if let Some(v) = cat.as_any().downcast_ref::<StringArray>() {
        for s in v.iter() {
            push(s);
        }
    } else {
        return Err(format!("unexpected cat type: {:?}", cat.data_type()).into());
    }
    Ok(())
}

fn count_alpha(col: &dyn ArrowArray) -> Result<usize, Box<dyn Error>> {
    if let Some(v) = col.as_any().downcast_ref::<StringViewArray>() {
        Ok(v.iter().filter(|s| *s == Some("alpha")).count())
    } else if let Some(v) = col.as_any().downcast_ref::<StringArray>() {
        Ok(v.iter().filter(|s| *s == Some("alpha")).count())
    } else {
        Err(format!("unexpected cat arrow type: {:?}", col.data_type()).into())
    }
}

fn build_and_write(
    session: &VortexSession,
    path: &str,
    rows: usize,
    chunk: usize,
    view: bool,
    ncats: usize,
) -> Result<(), Box<dyn Error>> {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int64, false),
        ArrowField::new("x", DataType::Int64, false),
        ArrowField::new("y", DataType::Float64, false),
        ArrowField::new("z", DataType::Float64, false),
        string_field("cat", view),
        string_field("tag", view),
    ]));
    let write_schema = session.arrow().from_arrow_schema(schema.as_ref())?;

    // UF_PAD: append this many 'x' chars to each string value to push it past the 12-byte
    // StringView inline threshold, forcing an actual variadic data buffer (like non-inline Arrow).
    let pad = env_usize("UF_PAD", 0);
    let suffix: String = "x".repeat(pad);
    // UF_BLOCK: StringView builder fixed block size; small values force many variadic data buffers.
    let block = env_usize("UF_BLOCK", 0);

    let mut chunks: Vec<vortex::array::ArrayRef> = Vec::new();
    let mut written = 0usize;
    while written < rows {
        let batch = std::cmp::min(chunk, rows - written);
        let mut id = Vec::with_capacity(batch);
        let mut x = Vec::with_capacity(batch);
        let mut y = Vec::with_capacity(batch);
        let mut z = Vec::with_capacity(batch);
        let mut cat: Vec<Option<String>> = Vec::with_capacity(batch);
        let mut tag: Vec<Option<String>> = Vec::with_capacity(batch);
        for i in 0..batch {
            let r = (written + i) as i64;
            id.push(r);
            x.push((r * 2654435761) % 1_000_000);
            y.push((r as f64) * 0.5);
            z.push((r as f64) * 0.25);
            cat.push(Some(format!("{}{}", CATS[(r as usize) % ncats], suffix)));
            if r % 10 == 0 {
                tag.push(None);
            } else {
                tag.push(Some(format!("{}{}", r, suffix)));
            }
        }
        let columns: Vec<Arc<dyn ArrowArray>> = vec![
            Arc::new(Int64Array::from(id)),
            Arc::new(Int64Array::from(x)),
            Arc::new(Float64Array::from(y)),
            Arc::new(Float64Array::from(z)),
            string_col(cat, view, block),
            string_col(tag, view, block),
        ];
        let record_batch = RecordBatch::try_new(schema.clone(), columns)?;
        let vortex_batch = session
            .arrow()
            .from_arrow_record_batch(record_batch, schema.as_ref())?;
        chunks.push(vortex_batch);
        written += batch;
    }

    let iter = ArrayIteratorAdapter::new(
        write_schema,
        chunks.into_iter().map(Ok::<_, VortexError>),
    );
    let file = std::fs::File::create(path)?;
    let summary = session
        .write_options()
        .blocking(&*RUNTIME)
        .write(file, iter)?;
    println!(
        "write: rows={} chunk={} view={} ncats={} -> file_rows={} bytes={}",
        rows,
        chunk,
        view,
        ncats,
        summary.row_count(),
        summary.size()
    );
    Ok(())
}

/// fullScan: project everything, materialize each chunk, count cat=="alpha".
fn fullscan_alpha_count(
    file: &VortexFile,
    session: &VortexSession,
) -> Result<(u64, u64), Box<dyn Error>> {
    let scan = apply_scan_knobs(file.scan()?.with_projection(root()));
    let stream = scan.into_array_stream()?;
    RUNTIME.block_on(async move {
        pin_mut!(stream);
        let mut ctx = session.create_execution_ctx();
        let mut rows: u64 = 0;
        let mut alpha: u64 = 0;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let arrow = session.arrow().execute_arrow(chunk, None, &mut ctx)?;
            rows += arrow.len() as u64;
            alpha += count_alpha_in_struct(&arrow)? as u64;
        }
        Ok::<(u64, u64), Box<dyn Error>>((rows, alpha))
    })
}

/// filter eq(col("cat"), lit("alpha")): count returned rows (and re-verify all are alpha).
fn filter_alpha_count(
    file: &VortexFile,
    session: &VortexSession,
) -> Result<(u64, u64), Box<dyn Error>> {
    let filter = eq(col("cat"), lit("alpha"));
    let scan = apply_scan_knobs(
        file.scan()?.with_projection(root()).with_some_filter(Some(filter)),
    );
    let stream = scan.into_array_stream()?;
    RUNTIME.block_on(async move {
        pin_mut!(stream);
        let mut ctx = session.create_execution_ctx();
        let mut rows: u64 = 0;
        let mut alpha: u64 = 0;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            let arrow = session.arrow().execute_arrow(chunk, None, &mut ctx)?;
            rows += arrow.len() as u64;
            alpha += count_alpha_in_struct(&arrow)? as u64;
        }
        Ok::<(u64, u64), Box<dyn Error>>((rows, alpha))
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let rows = env_usize("UF_ROWS", 2_000_000);
    let chunk = env_usize("UF_CHUNK", 65536);
    let view = env_usize("UF_VIEW", 1) != 0;
    let ncats = env_usize("UF_NCATS", 16);

    let keep = std::env::var("UF_KEEP").ok().filter(|s| !s.is_empty());
    // Read-only mode: skip writing and just read+diagnose an existing file (e.g. a Java-written one)
    // passed as argv[1] or via UF_READ.
    let read_only = std::env::args().nth(1).or_else(|| std::env::var("UF_READ").ok());
    let tmp = std::env::temp_dir().join(format!("underflow-{}.vortex", std::process::id()));
    let path = read_only
        .clone()
        .or_else(|| keep.clone())
        .unwrap_or_else(|| tmp.to_string_lossy().into_owned());

    println!("underflow: target_arch={ARCH} path={path} read_only={}", read_only.is_some());

    let session = VortexSession::default().with_handle(RUNTIME.handle());
    vortex_parquet_variant::initialize(&session);

    if read_only.is_none() {
        build_and_write(&session, &path, rows, chunk, view, ncats)?;
    }

    let file = RUNTIME.block_on(
        session
            .open_options()
            .with_layout_reader_cache()
            .open_path(&path),
    )?;
    let total = file.row_count();
    let expected = total / ncats as u64;

    if std::env::var("UF_SPLITS").ok().filter(|s| !s.is_empty()).is_some() {
        let splits = file.splits()?;
        println!("--- file.splits(): {} natural splits ---", splits.len());
        for (i, r) in splits.iter().enumerate().take(40) {
            println!("  split[{i}] = [{}, {})  len={}", r.start, r.end, r.end - r.start);
        }
    }

    let (fs_rows, fs_alpha) = fullscan_alpha_count(&file, &session)?;
    let (flt_rows, flt_alpha) = filter_alpha_count(&file, &session)?;

    if std::env::var("UF_POS").ok().filter(|s| !s.is_empty()).is_some() {
        // Assumes canonical Java file: row r has cat == CATS[r % ncats]. Reports positions where the
        // decoded value differs (the corrupted rows), and their distribution modulo the read batch.
        let stream = file.scan()?.with_projection(root()).into_array_stream()?;
        let session = session.clone();
        RUNTIME.block_on(async move {
            pin_mut!(stream);
            let mut ctx = session.create_execution_ctx();
            let mut bad: Vec<u64> = Vec::new();
            let mut batch_idx = 0usize;
            while let Some(chunk) = stream.next().await {
                let arrow = session.arrow().execute_arrow(chunk?, None, &mut ctx)?;
                let st = arrow.as_any().downcast_ref::<ArrowStructArray>().unwrap();
                let cat = st.column_by_name("cat").unwrap();
                let v = cat.as_any().downcast_ref::<StringViewArray>().unwrap();
                let id = st
                    .column_by_name("id")
                    .and_then(|c| c.as_any().downcast_ref::<Int64Array>().cloned())
                    .expect("id must be Int64");
                let batch_len = v.len();
                let id_first = id.value(0);
                let mut bad_in_batch = 0u64;
                let mut first_bad_at: i64 = -1;
                for (j, s) in v.iter().enumerate() {
                    let gr = id.value(j) as u64; // true row index
                    let expected = CATS[(gr as usize) % ncats];
                    if s != Some(expected) {
                        bad.push(gr);
                        bad_in_batch += 1;
                        if first_bad_at < 0 { first_bad_at = j as i64; }
                    }
                }
                if bad_in_batch > 0 {
                    println!("  emit#{batch_idx}: id_first={id_first} len={batch_len} cat_bad={bad_in_batch} first_bad_at={first_bad_at}");
                }
                batch_idx += 1;
            }
            println!("  total emitted batches={batch_idx}, total corrupted rows={}", bad.len());
            bad.sort_unstable();
            // Coalesce into contiguous runs of true row indices.
            let mut runs: Vec<(u64, u64)> = Vec::new();
            for &r in &bad {
                match runs.last_mut() {
                    Some(last) if r == last.1 + 1 => last.1 = r,
                    _ => runs.push((r, r)),
                }
            }
            println!("  corrupted contiguous runs (true row index): {} runs", runs.len());
            for (s, e) in runs.iter().take(20) {
                println!("    [{s}, {e}]  len={}", e - s + 1);
            }
            Ok::<(), Box<dyn Error>>(())
        })?;
    }

    if std::env::var("UF_HIST").ok().filter(|s| !s.is_empty()).is_some() {
        let stream = file.scan()?.with_projection(root()).into_array_stream()?;
        let session_a = session.clone();
        let (hist, nulls) = RUNTIME.block_on(async move {
            pin_mut!(stream);
            let mut ctx = session_a.create_execution_ctx();
            let mut hist = std::collections::BTreeMap::new();
            let mut nulls = 0u64;
            while let Some(chunk) = stream.next().await {
                let arrow = session_a.arrow().execute_arrow(chunk?, None, &mut ctx)?;
                tally_cat(&arrow, &mut hist, &mut nulls)?;
            }
            Ok::<_, Box<dyn Error>>((hist, nulls))
        })?;
        // Independent id integrity check: id should be the exact set {0..total}. Report sum + zeros.
        {
            let stream = file.scan()?.with_projection(root()).into_array_stream()?;
            let session2 = session.clone();
            let (id_sum, id_zeros, n) = RUNTIME.block_on(async move {
                pin_mut!(stream);
                let mut ctx = session2.create_execution_ctx();
                let (mut sum, mut zeros, mut n) = (0i128, 0u64, 0u64);
                while let Some(chunk) = stream.next().await {
                    let arrow = session2.arrow().execute_arrow(chunk?, None, &mut ctx)?;
                    let st = arrow.as_any().downcast_ref::<ArrowStructArray>().unwrap();
                    let id = st.column_by_name("id").unwrap().as_any().downcast_ref::<Int64Array>().unwrap().clone();
                    for j in 0..id.len() {
                        let v = id.value(j);
                        sum += v as i128;
                        if v == 0 { zeros += 1; }
                        n += 1;
                    }
                }
                Ok::<_, Box<dyn Error>>((sum, zeros, n))
            })?;
            let expected_sum: i128 = (0..total as i128).sum();
            println!("--- id integrity: n={n} sum={id_sum} expected_sum={expected_sum} match={} id==0 count={id_zeros} (expected 1) ---", id_sum == expected_sum);
        }
        println!("--- cat distribution (expected {} each of {} values) ---", rows / ncats, ncats);
        let mut sum = 0u64;
        for (k, v) in &hist {
            sum += *v;
            let delta = *v as i64 - (rows / ncats) as i64;
            println!("  {k:>10} = {v:>8}  (delta {delta:+})");
        }
        println!("  {:>10} = {nulls:>8}", "<null>");
        println!("  distinct_values={} sum(non-null)={sum} total_with_nulls={}", hist.len(), sum + nulls);
    }

    println!("total_rows={total} (scanned fullScan rows={fs_rows})");
    println!("expected_alpha={expected}");
    println!("fullScan_materialized_alpha={fs_alpha}");
    println!("filter_returned_rows={flt_rows} (of which cat==alpha: {flt_alpha})");

    let verdict = if flt_rows < fs_alpha {
        "READ/filter bug: filter drops real matches (filter < fullScan-materialized)"
    } else if fs_alpha < expected {
        "WRITE/decode bug: fullScan-materialized alpha below expected"
    } else {
        "NO REPRO: filter == fullScan == expected"
    };
    println!("VERDICT: {verdict}");

    if keep.is_none() && read_only.is_none() {
        drop(std::fs::remove_file(&path));
    }
    Ok(())
}
