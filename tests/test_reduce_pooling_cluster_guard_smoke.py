import json
from pathlib import Path

import pytest

from tests import _ac_large_helpers as ach
from tests import _diamond_benchmarks_helpers as dh
from tests import _sif4_large_helpers as sh


def test_reduce_pooling_cluster_guard_smoke(tmp_path: Path) -> None:
    if not (dh.cantera_available() and sh.cantera_available() and ach.cantera_available()):
        pytest.skip("cantera is required")

    diamond_root = tmp_path / "diamond"
    sif4_root = tmp_path / "sif4"
    ac_root = tmp_path / "ac"
    diamond_root.mkdir(parents=True, exist_ok=True)
    sif4_root.mkdir(parents=True, exist_ok=True)
    ac_root.mkdir(parents=True, exist_ok=True)

    diamond_trace = dh.build_diamond_trace(diamond_root, run_id="diamond_trace_cluster_guard")
    sif4_trace = sh.build_sif4_large_trace(sif4_root, run_id="sif4_trace_cluster_guard")
    ac_trace = ach.build_ac_large_trace(ac_root, run_id="ac_trace_cluster_guard")

    entries = [
        ("configs/reduce_diamond_benchmarks_pooling.yaml", dh.run_reduce_mode, diamond_root, diamond_trace),
        ("configs/reduce_sif4_benchmark_sin3n4_large_pooling.yaml", sh.run_reduce_mode, sif4_root, sif4_trace),
        ("configs/reduce_ac_benchmark_large_pooling.yaml", ach.run_reduce_mode, ac_root, ac_trace),
    ]
    for idx, (cfg_path, run_reduce_mode, root, trace_path) in enumerate(entries):
        summary_path = run_reduce_mode(
            tmp_path=root,
            config_path=cfg_path,
            trace_path=trace_path,
            run_id=f"pooling_cluster_guard_smoke_{idx}",
        )
        out = json.loads(summary_path.read_text())
        sel = dict(out.get("selected_metrics") or {})
        assert bool(out.get("gate_passed")) is True
        assert bool(sel.get("cluster_guard_passed", True)) is True
        assert float(sel.get("max_cluster_size_ratio", 0.0)) <= 0.45 + 1.0e-12
