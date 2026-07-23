# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "summarize-cold-io.py"


def load_module():
    spec = importlib.util.spec_from_file_location("summarize_cold_io", MODULE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_result(path: Path, runtime_ns: int) -> None:
    path.write_text(
        json.dumps(
            {
                "name": "fineweb_q07/datafusion:vortex-file-compressed",
                "value": runtime_ns,
                "all_runtimes": [runtime_ns],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_render_compares_cold_modes(tmp_path):
    module = load_module()
    write_result(tmp_path / "buffered-1.jsonl", 100_000_000)
    write_result(tmp_path / "buffered-2.jsonl", 120_000_000)
    write_result(tmp_path / "buffered-3.jsonl", 110_000_000)
    write_result(tmp_path / "direct-1.jsonl", 80_000_000)
    write_result(tmp_path / "direct-2.jsonl", 90_000_000)
    write_result(tmp_path / "direct-3.jsonl", 85_000_000)

    output = module.render(tmp_path, "7", None)

    assert "| Buffered | 3 | 110.000 | 100.000 | 120.000 |" in output
    assert "| `O_DIRECT` | 3 | 85.000 | 80.000 | 90.000 |" in output
    assert "**Direct / buffered: 0.773x (22.7% faster).**" in output
