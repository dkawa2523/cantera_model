import json
from pathlib import Path

import pytest

from tests._diamond_benchmarks_helpers import build_diamond_trace, cantera_available, run_reduce_mode


def test_reduce_diamond_balance_guard_smoke(tmp_path: Path) -> None:
    if not cantera_available():
        pytest.skip("cantera is required")

    trace_path = build_diamond_trace(tmp_path, run_id="diamond_trace_balance_guard")
    summary_path = run_reduce_mode(
        tmp_path=tmp_path,
        config_path="configs/reduce_diamond_benchmarks_pooling.yaml",
        trace_path=trace_path,
        run_id="reduce_diamond_benchmarks_pooling_balance_test",
    )
    out = json.loads(summary_path.read_text())
    sel = dict(out.get("selected_metrics") or {})
    ge = dict(out.get("gate_evidence") or {})
    assert "balance_metrics" in sel
    assert "balance_violations" in ge
