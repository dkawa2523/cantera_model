import json
from pathlib import Path

import pytest

from tests._ac_large_helpers import build_ac_large_trace, cantera_available, run_reduce_mode


def test_reduce_ac_large_physics_guard_smoke(tmp_path: Path) -> None:
    if not cantera_available():
        pytest.skip("cantera is required")

    trace_path = build_ac_large_trace(tmp_path, run_id="ac_trace_physics_guard")
    summary_path = run_reduce_mode(
        tmp_path=tmp_path,
        config_path="configs/reduce_ac_benchmark_large_pooling.yaml",
        trace_path=trace_path,
        run_id="reduce_ac_large_pooling_phys_test",
    )
    out = json.loads(summary_path.read_text())
    sel = dict(out["selected_metrics"])
    assert int(sel.get("floor_min_reactions", 0)) >= 12
    assert int(sel.get("floor_min_species", 0)) >= 8
    ge = dict(out.get("gate_evidence") or {})
    assert "floor_violations" in ge
    if not bool(sel.get("floor_passed", False)):
        assert bool(out.get("gate_passed", False)) is False
