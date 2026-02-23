import json
from pathlib import Path

import pytest

from tests._ac_large_helpers import build_ac_large_trace, cantera_available, run_reduce_mode


def test_reduce_ac_large_balance_v2_smoke(tmp_path: Path) -> None:
    if not cantera_available():
        pytest.skip("cantera is required")

    trace_path = build_ac_large_trace(tmp_path, run_id="ac_trace_balance_v2")
    summary_path = run_reduce_mode(
        tmp_path=tmp_path,
        config_path="configs/reduce_ac_benchmark_large_pooling.yaml",
        trace_path=trace_path,
        run_id="reduce_ac_large_pooling_balance_v2_test",
    )
    out = json.loads(summary_path.read_text())
    sel = dict(out.get("selected_metrics") or {})
    assert "weighted_active_species_coverage" in sel
    assert "essential_species_coverage" in sel
    assert str(sel.get("balance_mode", "")) == "hybrid"
