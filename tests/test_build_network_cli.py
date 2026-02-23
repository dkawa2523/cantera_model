import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from cantera_model.io.trace_store import save_case_bundle
from cantera_model.types import CaseBundle, CaseTrace


def _build_trace_bundle() -> CaseBundle:
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

    def make_case(case_id: str, t_end: float) -> CaseTrace:
        t = np.linspace(0.0, t_end, 7)
        rop = np.abs(
            np.vstack(
                [
                    np.linspace(1.0, 0.2, 7),
                    np.linspace(0.8, 0.4, 7),
                    np.linspace(0.6, 0.5, 7),
                    np.linspace(0.4, 0.9, 7),
                ]
            ).T
        )
        wdot = rop @ nu.T
        X = np.abs(np.cumsum(wdot, axis=0))
        X = X / np.maximum(X.sum(axis=1, keepdims=True), 1.0)
        temp = np.linspace(1000.0, 1150.0, 7)
        pressure = np.full(7, 101325.0)
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
        cases=[make_case("c0", 0.1), make_case("c1", 0.2)],
        meta={"species_meta": species_meta, "nu": nu.tolist()},
    )


def test_build_network_cli_from_trace(tmp_path) -> None:
    bundle = _build_trace_bundle()
    trace_path = tmp_path / "trace.h5"
    save_case_bundle(trace_path, bundle)

    out_root = tmp_path / "network"
    run_id = "n0"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.build_network",
            "--trace-h5",
            str(trace_path),
            "--run-id",
            run_id,
            "--output-root",
            str(out_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    out_dir = out_root / run_id
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["status"] == "ok"
    assert summary["species"] == 6
    assert summary["reactions"] == 4

    nu = np.load(out_dir / "nu.npy")
    A = np.load(out_dir / "A.npy")
    F = np.load(out_dir / "F_bar.npy")
    I = np.load(out_dir / "I_reaction.npy")
    rop = np.load(out_dir / "rop.npy")
    wdot = np.load(out_dir / "wdot.npy")
    dt = np.load(out_dir / "dt.npy")
    time = np.load(out_dir / "time.npy")
    X = np.load(out_dir / "X.npy")
    case_slices = json.loads((out_dir / "case_slices.json").read_text())
    F_phase = np.load(out_dir / "F_bar_by_phase.npy")
    I_phase = np.load(out_dir / "I_reaction_by_phase.npy")

    assert nu.shape == (6, 4)
    assert A.shape[1] == 6
    assert F.shape == (6, 6)
    assert I.shape == (4,)
    assert rop.shape[1] == 4
    assert wdot.shape[1] == 6
    assert dt.shape[0] == rop.shape[0]
    assert time.shape == (rop.shape[0],)
    assert X.shape == (rop.shape[0], 6)
    assert len(case_slices) == 2
    assert F_phase.shape == (1, 6, 6)
    assert I_phase.shape == (1, 4)
    assert np.all(F >= 0.0)
    assert summary["phases"] == 1
    assert summary["phase_names"] == ["all"]
    assert summary["state_saved"] is True


def test_build_network_cli_phase_fractions(tmp_path) -> None:
    bundle = _build_trace_bundle()
    trace_path = tmp_path / "trace.h5"
    save_case_bundle(trace_path, bundle)

    out_root = tmp_path / "network"
    run_id = "n1"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.build_network",
            "--trace-h5",
            str(trace_path),
            "--run-id",
            run_id,
            "--output-root",
            str(out_root),
            "--phase-fractions",
            "pulse:0.4,purge:0.6",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    out_dir = out_root / run_id
    phase_names = json.loads((out_dir / "phase_names.json").read_text())
    phase_fracs = json.loads((out_dir / "phase_fractions.json").read_text())
    F_phase = np.load(out_dir / "F_bar_by_phase.npy")
    I_phase = np.load(out_dir / "I_reaction_by_phase.npy")

    assert phase_names == ["pulse", "purge"]
    assert len(phase_fracs) == 2
    assert abs(sum(float(x) for x in phase_fracs) - 1.0) < 1.0e-12
    assert F_phase.shape == (2, 6, 6)
    assert I_phase.shape == (2, 4)
