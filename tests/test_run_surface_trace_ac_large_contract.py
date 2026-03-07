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


def test_run_surface_trace_ac_large_contract(tmp_path: Path) -> None:
    if cantera is None:
        pytest.skip("cantera is required")

    cfg = yaml.safe_load(Path("configs/ac_benchmark_large_trace.yaml").read_text())
    cfg["trace_output"] = {"root": str(tmp_path / "traces")}
    cfg["integration"] = {"n_steps": 12}
    cfg["conditions_csv"] = str(
        Path("cantera_model/benchmark_large/benchmarks/ac_hydrocarbon_cvd_large/conditions_quick.csv").resolve()
    )
    cfg["simulation"]["surface"]["mechanism"] = str(
        Path("cantera_model/benchmark_large/mechanisms/ac_hydrocarbon_cvd_large__gri30.yaml").resolve()
    )
    cfg_path = tmp_path / "trace_cfg_large.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    run_id = "ac_large_trace_contract"
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
    assert trace_path.exists()
    bundle = load_case_bundle(trace_path)

    assert bundle.phase == "ac_surf"
    assert len(bundle.cases) == 4
    assert len(bundle.species_names) >= 50
    assert len(bundle.reaction_eqs) >= 100

    first = bundle.cases[0]
    assert first.X.shape[1] == len(bundle.species_names)
    assert first.wdot.shape == first.X.shape
    assert first.rop.shape[1] == len(bundle.reaction_eqs)

    meta = bundle.meta
    assert isinstance(meta.get("species_meta"), list)
    assert meta.get("include_gas_reactions_in_trace") is True
    assert meta.get("trace_wdot_policy") == "stoich_consistent"

    nu = np.asarray(meta.get("nu"), dtype=float)
    assert nu.shape == (len(bundle.species_names), len(bundle.reaction_eqs))
