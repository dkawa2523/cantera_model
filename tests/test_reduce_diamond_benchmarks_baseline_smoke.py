import json
from pathlib import Path

import pytest

from tests._diamond_benchmarks_helpers import build_diamond_trace, cantera_available, run_reduce_mode


def test_reduce_diamond_benchmarks_baseline_smoke(tmp_path: Path) -> None:
    if not cantera_available():
        pytest.skip("cantera is required")

    trace_path = build_diamond_trace(tmp_path, run_id="diamond_trace_baseline")
    summary_path = run_reduce_mode(
        tmp_path=tmp_path,
        config_path="configs/reduce_diamond_benchmarks_baseline.yaml",
        trace_path=trace_path,
        run_id="reduce_diamond_benchmarks_baseline_test",
    )
    out = json.loads(summary_path.read_text())
    assert out["data_source"] == "trace_h5"
    assert out["hard_ban_violations"] == 0
    assert out["selected_metrics"]["reduction_mode"] == "baseline"
