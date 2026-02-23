import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


try:
    import cantera  # noqa: F401
except Exception:  # pragma: no cover - environment dependent
    cantera = None


def test_run_compare_backward_compat(tmp_path) -> None:
    if cantera is None:
        pytest.skip("cantera is required")

    run_id = "compat_compare"
    env = dict(os.environ)
    env["PYTHON_BIN"] = sys.executable

    proc = subprocess.run(
        [
            "bash",
            "run_compare.sh",
            "--run-id",
            run_id,
            "--conditions-csv",
            "assets/conditions/gri30_tiny.csv",
            "--n-steps",
            "30",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr

    out_dir = Path("runs") / run_id
    summary_path = out_dir / "summary.json"
    metrics_path = out_dir / "metrics.json"
    cmp_path = out_dir / "comparison_results.csv"

    assert summary_path.exists()
    assert metrics_path.exists()
    assert cmp_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "ok"
