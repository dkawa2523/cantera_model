import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from cantera_model.io.trace_store import load_case_bundle


try:
    import cantera  # noqa: F401
except Exception:  # pragma: no cover - environment dependent
    cantera = None


def test_run_surface_trace_stoich_wdot(tmp_path: Path) -> None:
    if cantera is None:
        pytest.skip("cantera is required")

    cfg = yaml.safe_load(Path("configs/diamond_benchmarks_diamond_large_trace.yaml").read_text())
    cfg["trace_output"] = {"root": str(tmp_path / "traces")}
    cfg["integration"] = {"n_steps": 8}
    cfg["conditions_csv"] = str(
        Path("cantera_model/benchmarks_diamond/benchmarks/diamond_cvd_large/conditions.csv").resolve()
    )
    cfg["simulation"]["surface"]["mechanism"] = str(
        Path("cantera_model/benchmarks_diamond/mechanisms/diamond_gri30_multisite.yaml").resolve()
    )
    cfg["simulation"]["surface"]["include_gas_reactions_in_trace"] = True
    cfg["simulation"]["surface"]["trace_wdot_policy"] = "stoich_consistent"
    cfg_path = tmp_path / "trace_cfg_stoich.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    run_id = "diamond_large_trace_stoich"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.run_surface_trace",
            "--config",
            str(cfg_path),
            "--run-id",
            run_id,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    trace_path = tmp_path / "traces" / f"{run_id}.h5"
    bundle = load_case_bundle(trace_path)
    nu = np.asarray(bundle.meta.get("nu"), dtype=float)

    for case in bundle.cases:
        reconstructed = np.asarray(case.rop, dtype=float) @ nu.T
        assert reconstructed.shape == np.asarray(case.wdot, dtype=float).shape
        assert np.allclose(reconstructed, case.wdot, rtol=1.0e-7, atol=1.0e-9)
