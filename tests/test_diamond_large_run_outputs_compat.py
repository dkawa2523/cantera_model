import json
import subprocess
import sys
from pathlib import Path

import pytest


try:
    import cantera  # noqa: F401
except Exception:  # pragma: no cover - environment dependent
    cantera = None


def test_diamond_large_run_outputs_compat() -> None:
    if cantera is None:
        pytest.skip("cantera is required")

    run_id = "test_diamond_benchmarks_diamond_large"
    proc = subprocess.run(
        [
            sys.executable,
            "run_cantera_eval.py",
            "--config",
            "configs/diamond_benchmarks_diamond_large_quick.yaml",
            "--run-id",
            run_id,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    out_dir = Path("runs") / run_id
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "baseline_results.csv").exists()
    assert (out_dir / "candidate_results.csv").exists()
    assert (out_dir / "comparison_results.csv").exists()

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["status"] == "ok"
    assert summary["simulation_mode"] == "surface_batch"
    assert summary["surface_phase"] == "diamond_100_multi"
    assert summary["comparison"]["cases"] == 4
    assert summary["baseline_counts"]["species"] >= 50
    assert summary["baseline_counts"]["reactions"] >= 100
