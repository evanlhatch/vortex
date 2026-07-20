# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""SQL benchmark declarations and CI profiles.

Edit this file when changing benchmark coverage. Matrix rendering lives in
``bench_orchestrator.matrix`` so workflow shape and benchmark coverage do not drift together.
"""

from .config import Benchmark, Engine, Format
from .matrix import (
    DEFAULTS,
    STANDARD,
    BenchmarkDef,
    BenchmarkGroup,
    Profile,
    Storage,
    all_targets,
    defaults,
    df,
    duck,
    pr_base_benchmarks,
    pr_defaults,
    pr_full,
)


def _tpch(
    scale_factor: float | int,
    storage: Storage,
    *,
    group: BenchmarkGroup,
    pr_base: bool,
    iterations: int | None = 10,
) -> BenchmarkDef:
    suffix = "" if scale_factor in {1, 100} else f"-{int(scale_factor)}"
    if storage is Storage.NVME:
        target_set = df(
            Format.ARROW,
            Format.PARQUET,
            Format.VORTEX,
            Format.VORTEX_COMPACT,
            Format.LANCE,
        ) | duck(Format.PARQUET, Format.VORTEX, Format.VORTEX_COMPACT, Format.DUCKDB)
        local_dir = None
        remote_key = None
    else:
        target_set = STANDARD
        local_dir = f"vortex-bench/data/tpch/{scale_factor:.1f}"
        remote_key = f"tpch/{scale_factor:.1f}"

    name = f"TPC-H on {storage.label}" if scale_factor == 100 else f"TPC-H SF={scale_factor:g} on {storage.label}"
    return BenchmarkDef(
        id=f"tpch-{storage.value}{suffix}",
        benchmark=Benchmark.TPCH,
        name=name,
        targets=target_set,
        storage=storage,
        scale_factor=scale_factor,
        iterations=iterations,
        group=group,
        pr_base=pr_base,
        local_dir=local_dir,
        remote_key=remote_key,
    )


def _clickbench(benchmark: Benchmark, name: str) -> BenchmarkDef:
    return BenchmarkDef(
        id=f"{benchmark.value}-nvme",
        benchmark=benchmark,
        name=name,
        targets=df(Format.PARQUET, Format.VORTEX, Format.VORTEX_COMPACT, Format.LANCE)
        | duck(Format.PARQUET, Format.VORTEX, Format.VORTEX_COMPACT, Format.DUCKDB),
        pr_targets=DEFAULTS | duck(Format.DUCKDB),
    )


def _fineweb(storage: Storage) -> BenchmarkDef:
    if storage is Storage.NVME:
        return BenchmarkDef(
            id="fineweb",
            benchmark=Benchmark.FINEWEB,
            name="FineWeb NVMe",
            targets=STANDARD,
            scale_factor=100,
        )
    return BenchmarkDef(
        id="fineweb-s3",
        benchmark=Benchmark.FINEWEB,
        name="FineWeb S3",
        targets=STANDARD,
        storage=Storage.S3,
        scale_factor=100,
        local_dir="vortex-bench/data/fineweb",
        remote_key="fineweb",
    )


BENCHMARKS: list[BenchmarkDef] = [
    _clickbench(Benchmark.CLICKBENCH, "Clickbench on NVME"),
    _clickbench(Benchmark.CLICKBENCH_SORTED, "Clickbench Sorted on NVME"),
    _tpch(1.0, Storage.NVME, group=BenchmarkGroup.REGULAR, pr_base=True),
    _tpch(1.0, Storage.S3, group=BenchmarkGroup.REGULAR, pr_base=True),
    _tpch(10.0, Storage.NVME, group=BenchmarkGroup.REGULAR, pr_base=True),
    _tpch(10.0, Storage.S3, group=BenchmarkGroup.REGULAR, pr_base=False),
    _tpch(100, Storage.NVME, group=BenchmarkGroup.NIGHTLY, pr_base=True, iterations=None),
    _tpch(100.0, Storage.S3, group=BenchmarkGroup.NIGHTLY, pr_base=True, iterations=None),
    BenchmarkDef(
        id="tpcds-nvme",
        benchmark=Benchmark.TPCDS,
        name="TPC-DS SF=1 on NVME",
        targets=STANDARD | duck(Format.DUCKDB),
        scale_factor=1.0,
    ),
    BenchmarkDef(
        id="statpopgen",
        benchmark=Benchmark.STATPOPGEN,
        name="Statistical and Population Genetics",
        targets=STANDARD.only(Engine.DUCKDB),
        scale_factor=100,
        local_dir="vortex-bench/data/statpopgen",
    ),
    _fineweb(Storage.NVME),
    _fineweb(Storage.S3),
    BenchmarkDef(
        id="polarsignals",
        benchmark=Benchmark.POLARSIGNALS,
        name="PolarSignals Profiling",
        targets=df(Format.VORTEX),
        scale_factor=1,
    ),
    BenchmarkDef(
        id="appian-nvme",
        benchmark=Benchmark.APPIAN,
        name="Appian on NVME",
        targets=STANDARD | duck(Format.DUCKDB),
        pr_targets=DEFAULTS | duck(Format.DUCKDB),
        pr_base=False,
        iterations=10,
    ),
    BenchmarkDef(
        id="vortex-queries",
        benchmark=Benchmark.VORTEX_QUERIES,
        name="Vortex queries",
        targets=DEFAULTS,
        group=BenchmarkGroup.VORTEX,
        iterations=100,
    ),
]

PROFILES: dict[str, Profile] = {
    "develop": Profile(
        targets=all_targets,
        description="Every regular SQL benchmark at full target coverage.",
    ),
    "pr": Profile(
        benchmarks=pr_base_benchmarks,
        targets=pr_defaults,
        description="The cheaper pull-request SQL benchmark lane.",
    ),
    "pr-full": Profile(
        targets=pr_full,
        data_formats=all_targets,
        description="Every regular SQL benchmark at full PR target coverage.",
    ),
    "nightly": Profile(
        group=BenchmarkGroup.NIGHTLY,
        targets=defaults,
        description="Large-scale SF=100 TPC-H on NVMe and S3 at default targets.",
    ),
    "vortex": Profile(
        group=BenchmarkGroup.VORTEX,
        description="The Vortex query suite run on pushes and pull requests.",
    ),
}
