import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


try:
    import cantera  # noqa: F401
except Exception:  # pragma: no cover - environment dependent
    cantera = None


def test_reduce_diamond_benchmarks_large_baseline_smoke(tmp_path: Path) -> None:
    if cantera is None:
        pytest.skip("cantera is required")

    trace_cfg = yaml.safe_load(Path("configs/diamond_benchmarks_diamond_large_trace.yaml").read_text())
    trace_cfg["trace_output"] = {"root": str(tmp_path / "traces")}
    trace_cfg["integration"] = {"n_steps": 8}
    trace_cfg["conditions_csv"] = str(
        Path("cantera_model/benchmarks_diamond/benchmarks/diamond_cvd_large/conditions.csv").resolve()
    )
    trace_cfg["simulation"]["surface"]["mechanism"] = str(
        Path("cantera_model/benchmarks_diamond/mechanisms/diamond_gri30_multisite.yaml").resolve()
    )
    trace_cfg_path = tmp_path / "trace_cfg.yaml"
    trace_cfg_path.write_text(yaml.safe_dump(trace_cfg, sort_keys=False))

    trace_run_id = "diamond_large_trace_for_reduce_smoke"
    trace_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.run_surface_trace",
            "--config",
            str(trace_cfg_path),
            "--run-id",
            trace_run_id,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert trace_proc.returncode == 0, trace_proc.stderr

    trace_path = tmp_path / "traces" / f"{trace_run_id}.h5"
    assert trace_path.exists()

    reduce_cfg = yaml.safe_load(Path("configs/reduce_diamond_benchmarks_large_baseline.yaml").read_text())
    reduce_cfg["trace_h5"] = str(trace_path)
    reduce_cfg["report_dir"] = str(tmp_path / "reports")
    reduce_cfg_path = tmp_path / "reduce_cfg.yaml"
    reduce_cfg_path.write_text(yaml.safe_dump(reduce_cfg, sort_keys=False))

    reduce_run_id = "reduce_diamond_large_baseline_smoke"
    reduce_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.reduce_validate",
            "--config",
            str(reduce_cfg_path),
            "--run-id",
            reduce_run_id,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert reduce_proc.returncode == 0, reduce_proc.stderr

    summary_path = tmp_path / "reports" / reduce_run_id / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())

    assert summary["data_source"] == "trace_h5"
    assert isinstance(summary.get("gate_passed"), bool)
    assert summary["selected_metrics"]["species_before"] >= 50
    assert summary["selected_metrics"]["reactions_before"] >= 100
