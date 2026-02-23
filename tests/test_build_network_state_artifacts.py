import json
import subprocess
import sys

import numpy as np

from cantera_model.io.trace_store import save_case_bundle
from cantera_model.types import CaseBundle, CaseTrace


def _bundle() -> CaseBundle:
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
    t = np.linspace(0.0, 0.2, 6)
    rop = np.abs(np.vstack([np.linspace(1.0, 0.2, 6), np.linspace(0.8, 0.4, 6), np.linspace(0.6, 0.5, 6), np.linspace(0.4, 0.9, 6)]).T)
    wdot = rop @ nu.T
    X = np.abs(np.cumsum(wdot, axis=0))
    X = X / np.maximum(X.sum(axis=1, keepdims=True), 1.0)
    case = CaseTrace(
        case_id="c0",
        time=t,
        temperature=np.linspace(1000.0, 1100.0, 6),
        pressure=np.full(6, 101325.0),
        X=X,
        wdot=wdot,
        rop=rop,
        species_names=species,
        reaction_eqs=reactions,
        meta={"conditions": {"case_id": "c0", "T0": 1000.0, "P0_atm": 1.0, "phi": 1.0, "t_end": 0.2}},
    )
    species_meta = [{"name": s, "composition": {"C": 1} if "C" in s else {"H": 1}, "phase": "gas", "charge": 0, "radical": False, "role": ""} for s in species]
    return CaseBundle(
        mechanism_path="dummy.yaml",
        phase="gri30",
        species_names=species,
        reaction_eqs=reactions,
        cases=[case],
        meta={"species_meta": species_meta, "nu": nu.tolist()},
    )


def test_build_network_state_artifacts_default(tmp_path) -> None:
    trace_path = tmp_path / "trace.h5"
    save_case_bundle(trace_path, _bundle())
    out_root = tmp_path / "network"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.build_network",
            "--trace-h5",
            str(trace_path),
            "--run-id",
            "n0",
            "--output-root",
            str(out_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out_dir = out_root / "n0"
    assert (out_dir / "time.npy").exists()
    assert (out_dir / "X.npy").exists()
    slices = json.loads((out_dir / "case_slices.json").read_text())
    assert len(slices) == 1


def test_build_network_state_artifacts_disable(tmp_path) -> None:
    trace_path = tmp_path / "trace.h5"
    save_case_bundle(trace_path, _bundle())
    out_root = tmp_path / "network"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.build_network",
            "--trace-h5",
            str(trace_path),
            "--run-id",
            "n1",
            "--output-root",
            str(out_root),
            "--save-state",
            "false",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out_dir = out_root / "n1"
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["state_saved"] is False
    assert not (out_dir / "time.npy").exists()
    assert not (out_dir / "X.npy").exists()
