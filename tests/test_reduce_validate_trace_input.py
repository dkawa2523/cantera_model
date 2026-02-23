import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from cantera_model.io.trace_store import save_case_bundle
from cantera_model.types import CaseBundle, CaseTrace


def _build_bundle() -> CaseBundle:
    species = ["CH4", "CH", "CF4", "F", "N", "H"]
    reactions = ["r0", "r1", "r2", "r3"]
    nu = np.array(
        [
            [-1.0, 0.0, 0.0, 1.0],
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    def case(case_id: str, t_end: float) -> CaseTrace:
        t = np.linspace(0.0, t_end, 8)
        rop = np.abs(np.vstack([np.linspace(1.0, 0.2, 8), np.linspace(0.8, 0.4, 8), np.linspace(0.6, 0.5, 8), np.linspace(0.4, 0.9, 8)]).T)
        wdot = rop @ nu.T
        X = np.abs(np.cumsum(wdot, axis=0))
        X = X / np.maximum(X.sum(axis=1, keepdims=True), 1.0)
        temp = 1000.0 + 50.0 * t
        pressure = np.full_like(t, 101325.0)
        return CaseTrace(
            case_id=case_id,
            time=t,
            temperature=temp,
            pressure=pressure,
            X=X,
            wdot=wdot,
            rop=rop,
            species_names=species,
            reaction_eqs=reactions,
            meta={"conditions": {"case_id": case_id, "T0": float(temp[0]), "P0_atm": 1.0, "phi": 1.0, "t_end": t_end}},
        )

    species_meta = [
        {"name": "CH4", "composition": {"C": 1, "H": 4}, "phase": "gas", "charge": 0, "radical": False, "role": "fuel"},
        {"name": "CH", "composition": {"C": 1, "H": 1}, "phase": "gas", "charge": 0, "radical": True, "role": "fuel"},
        {"name": "CF4", "composition": {"C": 1, "F": 4}, "phase": "gas", "charge": 0, "radical": False, "role": "etch"},
        {"name": "F", "composition": {"F": 1}, "phase": "gas", "charge": 0, "radical": True, "role": "etch"},
        {"name": "N", "composition": {"N": 1}, "phase": "gas", "charge": 0, "radical": True, "role": "carrier"},
        {"name": "H", "composition": {"H": 1}, "phase": "gas", "charge": 0, "radical": True, "role": "carrier"},
    ]

    return CaseBundle(
        mechanism_path="dummy.yaml",
        phase="gri30",
        species_names=species,
        reaction_eqs=reactions,
        cases=[case("c0", 0.1), case("c1", 0.2)],
        meta={"species_meta": species_meta, "nu": nu.tolist()},
    )


def test_reduce_validate_with_trace_h5(tmp_path) -> None:
    bundle = _build_bundle()
    trace_path = tmp_path / "trace.h5"
    save_case_bundle(trace_path, bundle)

    cfg = yaml.safe_load(Path("configs/reduce_surrogate_aggressive.yaml").read_text())
    cfg["report_dir"] = str(tmp_path / "reports")
    cfg["trace_h5"] = str(trace_path)
    cfg.pop("conditions_csv", None)

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.reduce_validate",
            "--config",
            str(cfg_path),
            "--run-id",
            "trace_mode",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    out = json.loads(proc.stdout)
    assert out["data_source"] == "trace_h5"
    assert out["hard_ban_violations"] == 0

    metrics_path = tmp_path / "reports" / "trace_mode" / "metrics.csv"
    with metrics_path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) >= 3
    assert all(int(r["hard_ban_violations"]) == 0 for r in rows)
