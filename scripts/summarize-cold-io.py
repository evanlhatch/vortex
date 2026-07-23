# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Summarize repeated cold buffered and direct-I/O benchmark runs as Markdown."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import NamedTuple


class Summary(NamedTuple):
    samples_ms: list[float]

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)

    @property
    def max_ms(self) -> float:
        return max(self.samples_ms)


def load_sample(path: Path) -> float:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("{"):
            records.append(json.loads(line))

    if len(records) != 1:
        raise ValueError(f"{path} contains {len(records)} benchmark records, expected exactly one")

    record = records[0]
    runtimes = record.get("all_runtimes")
    if not isinstance(runtimes, list) or len(runtimes) != 1:
        raise ValueError(f"{path} must contain exactly one runtime")

    return float(runtimes[0]) / 1_000_000


def load_mode(results_dir: Path, mode: str) -> Summary:
    paths = sorted(results_dir.glob(f"{mode}-*.jsonl"))
    if not paths:
        raise ValueError(f"no {mode} results found in {results_dir}")
    return Summary([load_sample(path) for path in paths])


def format_delta(ratio: float) -> str:
    percentage = abs(ratio - 1) * 100
    if ratio < 1:
        return f"{percentage:.1f}% faster"
    if ratio > 1:
        return f"{percentage:.1f}% slower"
    return "the same speed"


def render(results_dir: Path, query: str, filesystem: Path | None) -> str:
    buffered = load_mode(results_dir, "buffered")
    direct = load_mode(results_dir, "direct")
    if len(buffered.samples_ms) != len(direct.samples_ms):
        raise ValueError("buffered and direct modes must have the same number of samples")

    ratio = direct.median_ms / buffered.median_ms
    lines = [
        f"# Cold buffered vs direct I/O: FineWeb Q{query}",
        "",
        (
            "Each sample ran in a fresh process after Linux page caches were dropped. "
            "Buffered and direct runs alternated order."
        ),
        "",
        "| mode | samples | median (ms) | min (ms) | max (ms) |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| Buffered | {len(buffered.samples_ms)} | {buffered.median_ms:.3f} "
            f"| {buffered.min_ms:.3f} | {buffered.max_ms:.3f} |"
        ),
        (
            f"| `O_DIRECT` | {len(direct.samples_ms)} | {direct.median_ms:.3f} "
            f"| {direct.min_ms:.3f} | {direct.max_ms:.3f} |"
        ),
        "",
        f"**Direct / buffered: {ratio:.3f}x ({format_delta(ratio)}).**",
        "",
        "<details>",
        "<summary>Individual samples</summary>",
        "",
        f"- Buffered: {', '.join(f'{value:.3f} ms' for value in buffered.samples_ms)}",
        f"- `O_DIRECT`: {', '.join(f'{value:.3f} ms' for value in direct.samples_ms)}",
        "",
        "</details>",
    ]

    if filesystem is not None:
        lines.extend(
            [
                "",
                "<details>",
                "<summary>Filesystem</summary>",
                "",
                "```text",
                filesystem.read_text(encoding="utf-8").rstrip(),
                "```",
                "",
                "</details>",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--filesystem", type=Path)
    args = parser.parse_args()

    print(render(args.results_dir, args.query, args.filesystem), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
