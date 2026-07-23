<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Bloom skipping-index findings

This implementation answers the core questions in
[#8901](https://github.com/vortex-data/vortex/issues/8901): a custom aggregate can be persisted in a
`ZonedLayout`, rebound on a fresh reader, and used by an equality rewrite to skip zones without
changing scan planning.

Substring search is intentionally out of scope. `LIKE '%substr%'` needs a separate inverted-index
design rather than overloading the per-zone Bloom interface.

## Result

The Bloom skipping index consists of:

- a serializable per-zone aggregate stored as `Binary`,
- a scalar membership probe,
- an equality `StatsRewriteRule` that emits
  `not(bloom_contains(stat(root(), bloom), literal))`, and
- a `SkipIndex` bundle plus `ZonedLayoutOptions::with_skip_index` for explicit write-side
  declaration.

The roundtrip test writes four zones through the normal `WriteStrategyBuilder` pipeline, opens the
file with a fresh registered session, and checks both hits and misses. The hit keeps exactly one
zone. The absent value is deliberately inside every zone's min/max range, but the Bloom proof
skips all four zones. Indexed results are array-equal to a full scan through a fresh session that
does not know about the custom index.

Unknown readers remain safe when `allow_unknown()` is enabled: they ignore the unknown aggregate
and read the data child without pruning. #8904 provides unknown-aggregate ignorability and is now
on `develop`. #8905 is still open as of 2026-07-23, so its empty-zone-map reader bypass remains on
this branch as a prerequisite.

## Focused benchmark

Command:

```text
cargo test --release -p vortex-file --test bloom_skip_index bloom_point_lookup_benchmark -- --ignored --nocapture
```

The benchmark uses 2,097,152 interleaved, high-cardinality `i64` values in 256 zones of 8,192 rows.
Interleaving makes the point lookup fall inside every zone's min/max range, so ordinary range
statistics cannot skip a zone. Timings are medians of nine fresh-reader scans after two warmups.
Segment payload counts are measured below the file reader with a counting `SegmentSource`.

| Case | Zones pruned | File size | Median | Segment requests | Unique segments | Segment payload |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No Bloom | 0/256 | 16,845,924 B | 1.541 ms | 258 | 257 | 16,897,520 B |
| Bloom | 251/256 | 18,947,380 B | 0.571 ms | 7 | 6 | 2,504,196 B |

The Bloom scan was **2.70x faster** and moved **6.75x less segment payload**. Five zones survived:
the one true hit and four false positives. The file grew by 2,101,456 bytes, or 12.5%, from one
8 KiB filter per zone plus layout overhead.

The negative control uses cardinality 16 and queries a value present in every zone:

| Case | Zones pruned | File size | Median | Segment requests | Unique segments | Segment payload |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No Bloom | 0/256 | 16,843,876 B | 1.807 ms | 513 | 257 | 33,651,004 B |
| Bloom | 0/256 | 18,945,332 B | 1.832 ms | 513 | 257 | 35,752,396 B |

With nothing to prune, the indexed scan was 1.4% slower and moved 6.2% more segment payload. A
Bloom index only makes sense when avoided data reads pay for loading and probing the zone table.

## Existing SQL benchmark

The benchmark writer adds a Bloom index only to ClickBench `UserID`, using 8,192-row zones. It
retains the normal dictionary, compressor, coalescing, and flat-layout stages and writes the
candidate to a separate `vortex-file-bloom` directory. The candidate job times unchanged
DataFusion/Parquet alongside indexed DataFusion/Vortex so runner-wide movement is visible.

The earlier mixed-index run already isolated the point-lookup behavior: ClickBench q19 improved
from 29.23 ms to 21.42 ms, approximately **1.33x faster after its Parquet control**. One scan proved
12,294 of 12,299 zones absent and kept only five. The full 43-query suite stayed neutral because the
other queries could not use the equality index.

A Bloom-only controlled rerun is required to replace the mixed-index file-size number and confirm
the cleaned implementation's wall time.

## Interface recommendation

Keep the aggregate, scalar-function, and rewrite registries composable, but expose one bundle at
the registration and write-declaration boundary:

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

This keeps the implementation pieces reusable while giving callers one object to register on
writer and reader sessions and one explicit per-column declaration. The aggregate, probe, and
rewrite types are private behind `BloomSkipIndex`; users cannot accidentally register only part of
the index.

`WriteStrategyBuilder::with_field_zoned_options` is the per-column seam. It changes only the zoned
statistics for a field and preserves the normal physical data pipeline.

## Off-the-shelf filter primitive

Arrow Rust's `parquet::bloom_filter::Sbbf` provides stable split-block Bloom semantics, byte
serialization, membership checks, NDV/FPP sizing, and folding. It does not replace the
Vortex-specific aggregate persistence, stats rewrite, probe expression, or writer declaration.
Using it directly from `vortex-layout` would also add an undesirable Parquet dependency.

For production, put an SBBF-compatible implementation in a small neutral crate, then use it behind
`BloomSkipIndex`.

## Remaining work

- Aggregate identity includes serialized sizing options. Readers must currently register a rewrite
  with exactly the writer's byte and hash counts; production discovery should be automatic.
- The hash algorithm, format version, supported dtypes, null semantics, sizing policy, and
  saturation behavior need an explicit compatibility contract. This implementation supports
  `i64` equality only.
- Unknown aggregate fallback disables the whole zone map rather than retaining known statistics.
  That is safe but leaves pruning performance on the table.
- Filter sizing and zone length must remain workload-aware and opt-in. The low-cardinality control
  demonstrates the cost when pruning cannot occur.

Recommendation: continue with an opt-in equality Bloom index and the bundled `SkipIndex` interface.
The correctness proof, synthetic I/O reduction, and ClickBench point lookup justify a
production-quality follow-up. Design substring search separately as an inverted index.
