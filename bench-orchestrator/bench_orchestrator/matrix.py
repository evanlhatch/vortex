# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Resolve declarative benchmark definitions into GitHub Actions matrices."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

from .config import Benchmark, BenchmarkTarget, Engine, Format


class Storage(Enum):
    """Where a benchmark's data lives when it runs."""

    NVME = "nvme"
    S3 = "s3"

    @property
    def label(self) -> str:
        """Human-facing name used in benchmark display names."""
        return "NVME" if self is Storage.NVME else "S3"


class BenchmarkGroup(Enum):
    """A separately scheduled group of benchmark definitions."""

    REGULAR = "regular"
    NIGHTLY = "nightly"
    VORTEX = "vortex"


def _dedupe(targets: Iterable[BenchmarkTarget]) -> tuple[BenchmarkTarget, ...]:
    """Normalize and de-duplicate targets, preserving first-seen order."""
    return tuple(dict.fromkeys(target.normalized() for target in targets))


@dataclass(frozen=True)
class TargetSet:
    """An ordered set of engine/format targets with small set algebra."""

    targets: tuple[BenchmarkTarget, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", _dedupe(self.targets))

    def __or__(self, other: TargetSet) -> TargetSet:
        """Return the ordered union of two target sets."""
        return TargetSet(self.targets + other.targets)

    def only(self, *engines: Engine) -> TargetSet:
        """Restrict the target set to the given engines."""
        keep = set(engines)
        return TargetSet(tuple(target for target in self.targets if target.engine in keep))

    def formats(self) -> list[Format]:
        """Return referenced formats in first-seen order."""
        return list(dict.fromkeys(target.format for target in self.targets))

    def __iter__(self):
        return iter(self.targets)

    def __len__(self) -> int:
        return len(self.targets)


def targets(engine: Engine, *formats: Format) -> TargetSet:
    """Build targets for one engine across several formats."""
    return TargetSet(tuple(BenchmarkTarget(engine=engine, format=fmt) for fmt in formats))


def df(*formats: Format) -> TargetSet:
    """Build DataFusion targets across several formats."""
    return targets(Engine.DATAFUSION, *formats)


def duck(*formats: Format) -> TargetSet:
    """Build DuckDB targets across several formats."""
    return targets(Engine.DUCKDB, *formats)


STANDARD = df(Format.PARQUET, Format.VORTEX, Format.VORTEX_COMPACT) | duck(
    Format.PARQUET, Format.VORTEX, Format.VORTEX_COMPACT
)
DEFAULTS = df(Format.PARQUET, Format.VORTEX) | duck(Format.PARQUET, Format.VORTEX)
_NOT_GENERATED = frozenset({Format.ARROW, Format.LANCE})
_FORMAT_ORDER = (
    Format.ARROW,
    Format.PARQUET,
    Format.VORTEX,
    Format.VORTEX_COMPACT,
    Format.VORTEX_NATIVE,
    Format.DUCKDB,
    Format.LANCE,
)


@dataclass(frozen=True)
class BenchmarkDef:
    """A benchmark and the canonical target superset it can run."""

    id: str
    benchmark: Benchmark
    name: str
    targets: TargetSet
    storage: Storage = Storage.NVME
    scale_factor: float | int | None = None
    iterations: int | None = None
    group: BenchmarkGroup = BenchmarkGroup.REGULAR
    pr_targets: TargetSet | None = None
    pr_base: bool = True
    local_dir: str | None = None
    remote_key: str | None = None

    @property
    def subcommand(self) -> str:
        """Return the ``vx-bench`` subcommand for this benchmark."""
        return self.benchmark.value


TargetPolicy = Callable[[BenchmarkDef], TargetSet]
BenchmarkPolicy = Callable[[BenchmarkDef], bool]


def all_targets(benchmark: BenchmarkDef) -> TargetSet:
    """Run every target declared by a benchmark."""
    return benchmark.targets


def defaults(benchmark: BenchmarkDef) -> TargetSet:
    """Run the cheap default lane intersected with a benchmark's declared targets."""
    return TargetSet(tuple(target for target in benchmark.targets if target in DEFAULTS.targets))


def pr_full(benchmark: BenchmarkDef) -> TargetSet:
    """Return the full target set supported by pull-request runners."""
    if benchmark.pr_targets is not None:
        return benchmark.pr_targets
    return TargetSet(tuple(target for target in benchmark.targets if target.format is not Format.LANCE))


def pr_defaults(benchmark: BenchmarkDef) -> TargetSet:
    """Return the cheap PR lane while preserving Arrow coverage for regular TPC-H."""
    allowed = set(DEFAULTS)
    if benchmark.benchmark is Benchmark.TPCH:
        allowed.update(df(Format.ARROW))
    return TargetSet(tuple(target for target in pr_full(benchmark) if target in allowed))


def all_benchmarks(_: BenchmarkDef) -> bool:
    """Include every benchmark in a profile's group."""
    return True


def pr_base_benchmarks(benchmark: BenchmarkDef) -> bool:
    """Include benchmarks assigned to the cheaper PR lane."""
    return benchmark.pr_base


@dataclass(frozen=True)
class Profile:
    """A named CI benchmark configuration."""

    group: BenchmarkGroup = BenchmarkGroup.REGULAR
    benchmarks: BenchmarkPolicy = all_benchmarks
    targets: TargetPolicy = all_targets
    data_formats: TargetPolicy | None = None
    description: str = ""


def _valid_for_storage(target_set: TargetSet, storage: Storage) -> TargetSet:
    """Drop targets that are invalid for the storage backend."""
    if storage is Storage.S3:
        return TargetSet(tuple(target for target in target_set if target.format is not Format.LANCE))
    return target_set


def _data_formats(target_set: TargetSet) -> list[Format]:
    """Return data formats that the data-generation step must produce."""
    present = set(target_set.formats())
    return [fmt for fmt in _FORMAT_ORDER if fmt in present and fmt not in _NOT_GENERATED]


def _matrix_entry(benchmark: BenchmarkDef, run_targets: TargetSet, data_format_targets: TargetSet) -> dict[str, object]:
    """Build one GitHub Actions ``include`` entry."""
    entry: dict[str, object] = {
        "id": benchmark.id,
        "subcommand": benchmark.subcommand,
        "name": benchmark.name,
        "targets": [target.to_dict() for target in run_targets],
        "data_formats": [fmt.value for fmt in _data_formats(data_format_targets)],
    }
    if benchmark.scale_factor is not None:
        entry["scale_factor"] = str(benchmark.scale_factor)
    if benchmark.iterations is not None:
        entry["iterations"] = str(benchmark.iterations)
    if benchmark.local_dir is not None:
        entry["local_dir"] = benchmark.local_dir
    if benchmark.remote_key is not None:
        entry["remote_key"] = benchmark.remote_key
    return entry


def resolve_matrix(profile: Profile, benchmarks: Iterable[BenchmarkDef]) -> list[dict[str, object]]:
    """Resolve a profile into GitHub Actions matrix entries."""
    entries: list[dict[str, object]] = []
    for benchmark in benchmarks:
        if benchmark.group is not profile.group or not profile.benchmarks(benchmark):
            continue
        run_targets = _valid_for_storage(profile.targets(benchmark), benchmark.storage)
        if len(run_targets) == 0:
            continue
        data_format_targets = run_targets
        if profile.data_formats is not None:
            data_format_targets = _valid_for_storage(profile.data_formats(benchmark), benchmark.storage)
        entries.append(_matrix_entry(benchmark, run_targets, data_format_targets))
    return entries
