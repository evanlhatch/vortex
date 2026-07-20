# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Contract tests for the declarative CI benchmark matrix."""

import json
from typing import cast

from bench_orchestrator import cli as cli_module
from bench_orchestrator.benchmarks import BENCHMARKS, PROFILES
from bench_orchestrator.config import Benchmark, Engine, Format
from bench_orchestrator.matrix import (
    DEFAULTS,
    BenchmarkDef,
    BenchmarkGroup,
    Profile,
    Storage,
    all_targets,
    defaults,
    df,
    duck,
    pr_full,
    resolve_matrix,
)
from typer.testing import CliRunner

runner = CliRunner()


def _targets(entry: dict[str, object]) -> list[dict[str, str]]:
    return cast("list[dict[str, str]]", entry["targets"])


def test_default_policy_only_narrows_declared_targets() -> None:
    benchmark = BenchmarkDef(
        id="duckdb-only",
        benchmark=Benchmark.TPCH,
        name="DuckDB only",
        targets=duck(Format.VORTEX, Format.DUCKDB),
    )

    assert set(defaults(benchmark)) == set(duck(Format.VORTEX))


def test_resolver_emits_the_fields_consumed_by_the_workflow() -> None:
    benchmark = BenchmarkDef(
        id="remote",
        benchmark=Benchmark.TPCH,
        name="Remote",
        targets=df(Format.ARROW, Format.PARQUET, Format.LANCE, Format.VORTEX) | duck(Format.DUCKDB),
        storage=Storage.S3,
        scale_factor=1,
        iterations=10,
        local_dir="data/tpch",
        remote_key="tpch/1.0",
    )

    [entry] = resolve_matrix(Profile(targets=all_targets), [benchmark])

    assert entry == {
        "id": "remote",
        "subcommand": "tpch",
        "name": "Remote",
        "targets": [
            {"engine": "datafusion", "format": "arrow"},
            {"engine": "datafusion", "format": "parquet"},
            {"engine": "datafusion", "format": "vortex"},
            {"engine": "duckdb", "format": "duckdb"},
        ],
        "data_formats": ["parquet", "vortex", "duckdb"],
        "scale_factor": "1",
        "iterations": "10",
        "local_dir": "data/tpch",
        "remote_key": "tpch/1.0",
    }


def test_ci_profiles_have_distinct_and_consistent_roles() -> None:
    assert set(PROFILES) == {"develop", "pr", "pr-full", "nightly", "vortex"}
    regular_ids = {benchmark.id for benchmark in BENCHMARKS if benchmark.group is BenchmarkGroup.REGULAR}
    nightly_ids = {benchmark.id for benchmark in BENCHMARKS if benchmark.group is BenchmarkGroup.NIGHTLY}
    develop = {entry["id"]: entry for entry in resolve_matrix(PROFILES["develop"], BENCHMARKS)}
    pr = {entry["id"]: entry for entry in resolve_matrix(PROFILES["pr"], BENCHMARKS)}
    pr_full_matrix = {entry["id"]: entry for entry in resolve_matrix(PROFILES["pr-full"], BENCHMARKS)}
    nightly = {entry["id"]: entry for entry in resolve_matrix(PROFILES["nightly"], BENCHMARKS)}
    vortex = resolve_matrix(PROFILES["vortex"], BENCHMARKS)

    assert set(develop) == regular_ids
    assert set(pr_full_matrix) == regular_ids
    assert set(pr) == regular_ids - {"appian-nvme", "tpch-s3-10"}
    assert set(nightly) == nightly_ids
    assert [entry["id"] for entry in vortex] == ["vortex-queries"]
    assert develop["tpch-s3"]["remote_key"] == "tpch/1.0"
    assert nightly["tpch-s3"]["remote_key"] == "tpch/100.0"
    assert "${{" not in json.dumps([*develop.values(), *pr.values(), *pr_full_matrix.values(), *nightly.values()])

    default_targets = {(target.engine, target.format) for target in DEFAULTS}
    pr_targets = default_targets | {(Engine.DATAFUSION, Format.ARROW)}
    for entry in pr.values():
        targets = {(Engine(target["engine"]), Format(target["format"])) for target in _targets(entry)}
        assert targets
        assert targets <= pr_targets
    for entry in nightly.values():
        targets = {(Engine(target["engine"]), Format(target["format"])) for target in _targets(entry)}
        assert targets
        assert targets <= default_targets

    tpch = develop["tpch-nvme"]
    assert [(target["engine"], target["format"]) for target in _targets(tpch)] == [
        ("datafusion", "arrow"),
        ("datafusion", "parquet"),
        ("datafusion", "vortex"),
        ("datafusion", "vortex-compact"),
        ("datafusion", "lance"),
        ("duckdb", "parquet"),
        ("duckdb", "vortex"),
        ("duckdb", "vortex-compact"),
        ("duckdb", "duckdb"),
    ]

    assert [(target["engine"], target["format"]) for target in _targets(pr_full_matrix["clickbench-nvme"])] == [
        ("datafusion", "parquet"),
        ("datafusion", "vortex"),
        ("duckdb", "parquet"),
        ("duckdb", "vortex"),
        ("duckdb", "duckdb"),
    ]
    assert pr_full_matrix["clickbench-nvme"]["data_formats"] == [
        "parquet",
        "vortex",
        "vortex-compact",
        "duckdb",
    ]
    assert [(target["engine"], target["format"]) for target in _targets(pr["tpch-nvme"])] == [
        ("datafusion", "arrow"),
        ("datafusion", "parquet"),
        ("datafusion", "vortex"),
        ("duckdb", "parquet"),
        ("duckdb", "vortex"),
    ]
    assert all(target["format"] != "lance" for entry in pr_full_matrix.values() for target in _targets(entry))


def test_existing_display_and_scale_values_are_preserved() -> None:
    develop = {entry["id"]: entry for entry in resolve_matrix(PROFILES["develop"], BENCHMARKS)}
    nightly = {entry["id"]: entry for entry in resolve_matrix(PROFILES["nightly"], BENCHMARKS)}

    assert develop["tpch-nvme"]["scale_factor"] == "1.0"
    assert develop["statpopgen"]["scale_factor"] == "100"
    assert develop["polarsignals"]["scale_factor"] == "1"
    assert nightly["tpch-nvme"]["name"] == "TPC-H on NVME"
    assert nightly["tpch-nvme"]["scale_factor"] == "100"
    assert nightly["tpch-s3"]["name"] == "TPC-H on S3"
    assert nightly["tpch-s3"]["scale_factor"] == "100.0"


def test_matrix_command_emits_json_and_rejects_unknown_profiles() -> None:
    result = runner.invoke(cli_module.app, ["matrix", "develop"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)

    result = runner.invoke(cli_module.app, ["matrix", "does-not-exist"])
    assert result.exit_code == 1


def test_pr_full_policy_honors_benchmark_override() -> None:
    benchmark = BenchmarkDef(
        id="override",
        benchmark=Benchmark.APPIAN,
        name="Override",
        targets=df(Format.PARQUET, Format.VORTEX, Format.LANCE),
        pr_targets=df(Format.VORTEX),
    )

    assert pr_full(benchmark) == df(Format.VORTEX)
