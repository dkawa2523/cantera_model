import json
from pathlib import Path

import pytest

from tests._ac_large_helpers import build_ac_large_trace, cantera_available, run_reduce_mode


def test_reduce_ac_large_pooling_smoke(tmp_path: Path) -> None:
    if not cantera_available():
        pytest.skip("cantera is required")

    trace_path = build_ac_large_trace(tmp_path, run_id="ac_trace_pooling")
    summary_path = run_reduce_mode(
        tmp_path=tmp_path,
        config_path="configs/reduce_ac_benchmark_large_pooling.yaml",
        trace_path=trace_path,
        run_id="reduce_ac_large_pooling_test",
    )
    out = json.loads(summary_path.read_text())
    assert out["data_source"] == "trace_h5"
    assert out["hard_ban_violations"] == 0
    assert out["selected_metrics"]["reduction_mode"] in {"pooling", "baseline_fallback"}
