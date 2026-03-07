import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from cantera_model.io.trace_store import save_case_bundle
from cantera_model.types import CaseBundle, CaseTrace


def _bundle() -> CaseBundle:
    species = ["CH4", "CH", "CF4", "F", "N", "H"]
    reactions = ["r0", "r1", "r2", "r3"]
    nu = np.array(
        [
            [-1.0, 1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 1.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    def one(case_id: str, t_end: float) -> CaseTrace:
        t = np.linspace(0.0, t_end, 7)
        rop = np.abs(np.vstack([np.linspace(1.0, 0.2, 7), np.linspace(0.8, 0.4, 7), np.linspace(0.6, 0.5, 7), np.linspace(0.4, 0.9, 7)]).T)
        wdot = rop @ nu.T
        X = np.abs(np.cumsum(wdot, axis=0))
        X = X / np.maximum(X.sum(axis=1, keepdims=True), 1.0)
        return CaseTrace(
            case_id=case_id,
            time=t,
            temperature=np.linspace(1000.0, 1150.0, 7),
            pressure=np.full(7, 101325.0),
            X=X,
            wdot=wdot,
            rop=rop,
            species_names=species,
            reaction_eqs=reactions,
            meta={"conditions": {"case_id": case_id, "T0": 1000.0, "P0_atm": 1.0, "phi": 1.0, "t_end": t_end}},
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
        cases=[one("c0", 0.1), one("c1", 0.2)],
        meta={"species_meta": species_meta, "nu": nu.tolist()},
    )


def test_physical_gate_from_trace_h5(tmp_path) -> None:
    trace_path = tmp_path / "trace.h5"
    save_case_bundle(trace_path, _bundle())

    cfg = yaml.safe_load(Path("configs/reduce_surrogate_aggressive.yaml").read_text())
    cfg["trace_h5"] = str(trace_path)
    cfg.pop("conditions_csv", None)
    cfg["report_dir"] = str(tmp_path / "reports")
    eval_cfg = dict(cfg.get("evaluation") or {})
    eval_cfg["physical_gate"] = {"enabled": True, "max_conservation_violation": 1.0e6, "max_negative_steps": 10000}
    eval_cfg["surrogate_split"] = {"mode": "in_sample"}
    cfg["evaluation"] = eval_cfg

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
            "physical_trace",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["gate_evidence"]["physical_gate_enabled"] is True
    selected = out["gate_evidence"]["selected_stage_evidence"]
    assert selected["trajectory_steps"] > 0
    assert selected["state_source"] in {"trace_h5", "network_artifacts", "trace_h5_fallback", "wdot_reconstructed"}

    metrics_path = tmp_path / "reports" / "physical_trace" / "metrics.csv"
    with metrics_path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all("conservation_violation" in r for r in rows)
    assert all("negative_steps" in r for r in rows)
