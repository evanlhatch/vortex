<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Experimental skipping-index findings

This spike answers the core questions in [#8901](https://github.com/vortex-data/vortex/issues/8901): a third-party-style aggregate can be persisted in a `ZonedLayout`, rebound on a fresh reader, and used by an equality rewrite to skip zones without changing scan planning.

## Result

Yes, it works. The prototype adds experimental `i64` equality and UTF-8 trigram-bloom indexes consisting of:

- a serializable per-zone `BloomFilter` aggregate stored as `Binary`,
- a `bloom_contains` scalar probe,
- an equality `StatsRewriteRule` that emits `not(bloom_contains(stat(root(), bloom), literal))`, and
- a small `SkipIndex` bundle plus `ZonedLayoutOptions::with_skip_index` as the explicit write-side declaration.

The trigram variant stores every three-byte window from a zone and rewrites case-sensitive
`LIKE` expressions when their pattern contains at least one three-byte literal run. For
`LIKE '%needle%'`, a zone is skipped if any required trigram is definitely absent. SQL `%` and
`_` wildcards and backslash escapes are parsed conservatively; `ILIKE`, negated `LIKE`, and patterns
without a three-byte literal remain inconclusive.

The roundtrip tests write four zones, open the files with fresh registered sessions, and check both
hits and misses. The equality hit keeps exactly one data zone. Its absent value is deliberately
inside every zone's min/max range, but the bloom proof skips all four zones. The substring hit also
keeps exactly one zone and an absent substring skips all four. Results from every indexed scan are
array-equal to scans through fresh sessions without the custom indexes.

The unregistered reader case also works when `allow_unknown()` is enabled: it ignores the unknown aggregate and reads the data without pruning. That behavior depends on #8904 and #8905. As of 2026-07-22 both PRs are open and absent from `develop`, so their commits are included on this experimental branch as prerequisites.

## Benchmark

Command:

```text
cargo test --release -p vortex-file --test experimental_bloom bloom_point_lookup_benchmark -- --ignored --nocapture
```

The focused benchmark uses 2,097,152 interleaved, high-cardinality `i64` values in 256 zones of 8,192 rows. Interleaving makes the point lookup fall inside every zone's min/max range, so ordinary range statistics cannot skip a zone. Timings are medians of nine fresh-reader scans after two warmups. Segment payload counts are measured below the file reader with a counting `SegmentSource`.

| Case | Zones pruned | File size | Median | Segment requests | Unique segments | Segment payload |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No bloom | 0/256 | 16,845,924 B | 1.541 ms | 258 | 257 | 16,897,520 B |
| Bloom | 251/256 | 18,947,380 B | 0.571 ms | 7 | 6 | 2,504,196 B |

The bloom scan was **2.70x faster** and moved **6.75x less segment payload**. A preceding clean run measured 2.57x, giving a 2.57-2.70x repeat range. Five zones survived: the one true hit and four bloom false positives. The file grew by 2,101,456 bytes, or 12.5%, from one 8 KiB bloom per zone plus layout overhead.

The negative control uses cardinality 16 and queries a value present in every zone:

| Case | Zones pruned | File size | Median | Segment requests | Unique segments | Segment payload |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No bloom | 0/256 | 16,843,876 B | 1.807 ms | 513 | 257 | 33,651,004 B |
| Bloom | 0/256 | 18,945,332 B | 1.832 ms | 513 | 257 | 35,752,396 B |

With nothing to prune, the indexed scan was 1.4% slower (0.986x) and caused 6.2% more segment payload. A bloom only makes sense when avoided data reads pay for loading and probing the zone table; it does not help low-cardinality or non-selective equality predicates.

These are in-memory buffer results from a focused release test, not storage-system throughput claims. Remote or high-latency storage should amplify the request reduction, while a very fast cached scan or smaller zones may make zone-map overhead more visible.

### Existing SQL suites

The benchmark branch also wires the indexes into the normal compressed writer without replacing
its dictionary, compressor, coalescing, or flat-layout stages:

- ClickBench `UserID`: equality bloom, 8,192-row zones.
- ClickBench `URL` and `Title`: 16 KiB trigram blooms, 8,192-row zones.
- FineWeb `url`, `text`, and `file_path`: 16 KiB trigram blooms, 1,024-row zones.

The benchmark artifact is written to a separate `vortex-file-skipping` directory so cached normal
Vortex files cannot contaminate the comparison. The focused draft-PR run covers ClickBench,
ClickBench sorted, FineWeb NVMe, and FineWeb S3 with DataFusion/Vortex. Results will be recorded here
after the dedicated benchmark run completes.

### Off-the-shelf pieces

Arrow Rust's `parquet::bloom_filter::Sbbf` is a credible production filter primitive: it has stable
split-block Bloom semantics, byte serialization, membership checks, NDV/FPP sizing, and folding.
It does not replace the Vortex-specific aggregate persistence, stats rewrite, probe expression, or
per-column writer declaration, and using it directly from `vortex-layout` would add an undesirable
Parquet dependency to the layout crate. The prototype therefore keeps a small mergeable bitset for
the experiment. For production, put an SBBF-compatible filter in a small neutral crate (or extract
the Arrow implementation) and reuse it for both equality and n-gram indexes. Pulling in a search
engine or tokenizer such as Tantivy only to enumerate byte trigrams would be substantially heavier
than the operation itself.

## Interface recommendation

Keep the aggregate, scalar function, and rewrite registries composable, but add a single bundle at the user-facing registration and write-declaration boundary:

```rust
pub trait SkipIndex: Debug + Send + Sync + 'static {
    fn aggregate_fn(&self, input_dtype: &DType) -> Option<AggregateFnRef>;
    fn register(&self, session: &VortexSession);
}

impl ZonedLayoutOptions {
    pub fn with_skip_index<I: SkipIndex + ?Sized>(
        self,
        index: &I,
        input_dtype: &DType,
        session: &VortexSession,
    ) -> VortexResult<Self>;
}
```

This is preferable to replacing the existing extension points with one large vtable. An index
author can still reuse built-in aggregates or probes, while callers have one object to register on
writer and reader sessions and one explicit per-column declaration. The prototype adds
`WriteStrategyBuilder::with_field_zoned_options` as the narrow per-column seam, retaining the normal
data pipeline instead of requiring callers to reconstruct it.

## Awkward parts and next steps

- Aggregate identity includes serialized options. The reader must register a rewrite using exactly the same bloom sizing and hash count as the writer; a mismatched index safely becomes inconclusive, but discovery should be automatic in a production API.
- Choosing filter size, hash count, gram size, and zone length is workload-specific. In particular,
  trigrams for large text zones can saturate, while very small zones impose substantial file-size
  and metadata overhead.
- Unknown aggregate fallback currently disables that zone map rather than using the known statistics alongside it. That is safe but leaves pruning performance on the table.
- The hash algorithm, format version, supported dtypes, null semantics, sizing policy, and saturation behavior need an explicit compatibility contract. This spike supports `i64` equality and byte-oriented UTF-8 trigrams only.
- Bloom stats for all zones are regular child-array data. Their layout and caching should be profiled for many indexed columns and remote reads.

Recommendation: pursue the bundle interface and an explicit per-column writer declaration, while retaining the three lower-level registries. The selective equality benchmark is large enough to justify a production-quality experiment, but index selection must remain workload-aware and opt-in. The SQL run must show that real substring selectivity survives filter saturation and pays back the added file bytes before the trigram prototype is considered a win.
