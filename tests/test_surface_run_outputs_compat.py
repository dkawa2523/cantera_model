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


def test_surface_run_outputs_compat(tmp_path: Path) -> None:
    if cantera is None:
        pytest.skip("cantera is required")

    conditions_csv = tmp_path / "diamond_onecase.csv"
    conditions_csv.write_text(
        "case_id,T_K,P_atm,composition,t_end_s,n_steps,area\n"
        "base,1200,0.0263,\"H:2e-03, H2:0.988, CH3:2e-04, CH4:0.01\",0.01,10,1.0\n"
    )

    run_id = "surface_compat_onecase"
    cfg = {
        "run_id": run_id,
        "simulation": {
            "mode": "surface_batch",
            "surface": {
                "interface_phase": "diamond_100",
                "gas_phase": "gas",
                "pressure_unit": "atm",
            },
        },
        "baseline": {"mechanism": "diamond.yaml"},
        "candidate": {"mechanism": "diamond.yaml"},
        "conditions_csv": str(conditions_csv),
        "integration": {"n_steps": 20},
        "evaluation": {
            "rel_tolerance": 0.25,
            "rel_eps": 1.0e-12,
            "qoi": {"selectors": ["gas_X:H2:final", "surface_theta:c6HH:final"]},
        },
        "output": {"root": "runs"},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    proc = subprocess.run(
        [sys.executable, "run_cantera_eval.py", "--config", str(cfg_path), "--run-id", run_id],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    out_dir = cfg_path.parent / "runs" / run_id
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "baseline_results.csv").exists()
    assert (out_dir / "candidate_results.csv").exists()
    assert (out_dir / "comparison_results.csv").exists()

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["status"] == "ok"
    assert summary["simulation_mode"] == "surface_batch"
