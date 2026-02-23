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
            [-1.0, 0.0, 0.0, 1.0],
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    def one(case_id: str, t_end: float) -> CaseTrace:
        t = np.linspace(0.0, t_end, 6)
        rop = np.abs(np.vstack([np.linspace(1.0, 0.2, 6), np.linspace(0.8, 0.4, 6), np.linspace(0.6, 0.5, 6), np.linspace(0.4, 0.9, 6)]).T)
        wdot = rop @ nu.T
        X = np.abs(np.cumsum(wdot, axis=0))
        X = X / np.maximum(X.sum(axis=1, keepdims=True), 1.0)
        return CaseTrace(
            case_id=case_id,
            time=t,
            temperature=np.linspace(1000.0, 1100.0, 6),
            pressure=np.full(6, 101325.0),
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


def test_reduce_validate_with_network_dir(tmp_path) -> None:
    trace_path = tmp_path / "trace.h5"
    save_case_bundle(trace_path, _bundle())

    net_root = tmp_path / "network"
    run_id = "net0"
    p_build = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.build_network",
            "--trace-h5",
            str(trace_path),
            "--run-id",
            run_id,
            "--output-root",
            str(net_root),
            "--phase-fractions",
            "pulse:0.5,purge:0.5",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_build.returncode == 0, p_build.stderr

    cfg = yaml.safe_load(Path("configs/reduce_surrogate_aggressive.yaml").read_text())
    cfg["report_dir"] = str(tmp_path / "reports")
    cfg["network_dir"] = str(net_root / run_id)
    cfg.pop("trace_h5", None)
    eval_cfg = dict(cfg.get("evaluation") or {})
    eval_cfg["physical_gate"] = {"enabled": True, "max_conservation_violation": 1.0e6, "max_negative_steps": 10000}
    eval_cfg["surrogate_split"] = {"mode": "in_sample"}
    cfg["evaluation"] = eval_cfg
    merge_cfg = dict(cfg.get("merge") or {})
    merge_cfg["phase_weights"] = {"pulse": 1.0, "purge": 0.0}
    cfg["merge"] = merge_cfg

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    p_reduce = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.reduce_validate",
            "--config",
            str(cfg_path),
            "--run-id",
            "r0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_reduce.returncode == 0, p_reduce.stderr

    out = json.loads(p_reduce.stdout)
    assert out["data_source"] == "network_dir"
    assert out["hard_ban_violations"] == 0
    assert out["phase_context"]["mode"] == "weights"
    assert out["phase_context"]["applied"] is True
    assert out["phase_context"]["phase_names"] == ["pulse", "purge"]
    assert out["phase_context"]["n_phase"] == 2
    assert out["phase_context"]["source"] == "F_bar_by_phase.npy"
    assert out["gate_evidence"]["physical_gate_enabled"] is True


def test_reduce_validate_with_network_dir_phase_select_name(tmp_path) -> None:
    trace_path = tmp_path / "trace.h5"
    save_case_bundle(trace_path, _bundle())

    net_root = tmp_path / "network"
    run_id = "net1"
    p_build = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.build_network",
            "--trace-h5",
            str(trace_path),
            "--run-id",
            run_id,
            "--output-root",
            str(net_root),
            "--phase-fractions",
            "pulse:0.5,purge:0.5",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_build.returncode == 0, p_build.stderr

    cfg = yaml.safe_load(Path("configs/reduce_surrogate_aggressive.yaml").read_text())
    cfg["report_dir"] = str(tmp_path / "reports")
    cfg["network_dir"] = str(net_root / run_id)
    cfg.pop("trace_h5", None)
    merge_cfg = dict(cfg.get("merge") or {})
    merge_cfg.pop("phase_weights", None)
    merge_cfg["phase_select"] = "purge"
    cfg["merge"] = merge_cfg

    cfg_path = tmp_path / "cfg_select_name.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    p_reduce = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.reduce_validate",
            "--config",
            str(cfg_path),
            "--run-id",
            "r_select_name",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_reduce.returncode == 0, p_reduce.stderr
    out = json.loads(p_reduce.stdout)
    assert out["phase_context"]["mode"] == "select"
    assert out["phase_context"]["selected_phase"] == "purge"
    assert out["phase_context"]["weights"] == [0.0, 1.0]


def test_reduce_validate_with_network_dir_phase_select_index(tmp_path) -> None:
    trace_path = tmp_path / "trace.h5"
    save_case_bundle(trace_path, _bundle())

    net_root = tmp_path / "network"
    run_id = "net2"
    p_build = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.build_network",
            "--trace-h5",
            str(trace_path),
            "--run-id",
            run_id,
            "--output-root",
            str(net_root),
            "--phase-fractions",
            "pulse:0.5,purge:0.5",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_build.returncode == 0, p_build.stderr

    cfg = yaml.safe_load(Path("configs/reduce_surrogate_aggressive.yaml").read_text())
    cfg["report_dir"] = str(tmp_path / "reports")
    cfg["network_dir"] = str(net_root / run_id)
    cfg.pop("trace_h5", None)
    merge_cfg = dict(cfg.get("merge") or {})
    merge_cfg.pop("phase_weights", None)
    merge_cfg["phase_select"] = 0
    cfg["merge"] = merge_cfg

    cfg_path = tmp_path / "cfg_select_idx.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    p_reduce = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.reduce_validate",
            "--config",
            str(cfg_path),
            "--run-id",
            "r_select_idx",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_reduce.returncode == 0, p_reduce.stderr
    out = json.loads(p_reduce.stdout)
    assert out["phase_context"]["mode"] == "select"
    assert out["phase_context"]["selected_phase"] == "pulse"
    assert out["phase_context"]["weights"] == [1.0, 0.0]


def test_reduce_validate_phase_select_and_weights_conflict(tmp_path) -> None:
    trace_path = tmp_path / "trace.h5"
    save_case_bundle(trace_path, _bundle())

    net_root = tmp_path / "network"
    run_id = "net3"
    p_build = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.build_network",
            "--trace-h5",
            str(trace_path),
            "--run-id",
            run_id,
            "--output-root",
            str(net_root),
            "--phase-fractions",
            "pulse:0.5,purge:0.5",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_build.returncode == 0, p_build.stderr

    cfg = yaml.safe_load(Path("configs/reduce_surrogate_aggressive.yaml").read_text())
    cfg["report_dir"] = str(tmp_path / "reports")
    cfg["network_dir"] = str(net_root / run_id)
    cfg.pop("trace_h5", None)
    merge_cfg = dict(cfg.get("merge") or {})
    merge_cfg["phase_select"] = "pulse"
    merge_cfg["phase_weights"] = {"pulse": 1.0, "purge": 0.0}
    cfg["merge"] = merge_cfg

    cfg_path = tmp_path / "cfg_conflict.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    p_reduce = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.reduce_validate",
            "--config",
            str(cfg_path),
            "--run-id",
            "r_conflict",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_reduce.returncode != 0
    assert "cannot be set at the same time" in (p_reduce.stderr + p_reduce.stdout)


def test_reduce_validate_network_dir_missing_state_artifacts_fallback(tmp_path) -> None:
    trace_path = tmp_path / "trace.h5"
    save_case_bundle(trace_path, _bundle())

    net_root = tmp_path / "network"
    run_id = "net_missing_state"
    p_build = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.build_network",
            "--trace-h5",
            str(trace_path),
            "--run-id",
            run_id,
            "--output-root",
            str(net_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_build.returncode == 0, p_build.stderr
    net_dir = net_root / run_id
    (net_dir / "time.npy").unlink()
    (net_dir / "X.npy").unlink()

    cfg = yaml.safe_load(Path("configs/reduce_surrogate_aggressive.yaml").read_text())
    cfg["report_dir"] = str(tmp_path / "reports")
    cfg["network_dir"] = str(net_dir)
    cfg.pop("trace_h5", None)
    eval_cfg = dict(cfg.get("evaluation") or {})
    eval_cfg["physical_gate"] = {"enabled": True, "max_conservation_violation": 1.0e6, "max_negative_steps": 10000}
    eval_cfg["surrogate_split"] = {"mode": "in_sample"}
    cfg["evaluation"] = eval_cfg

    cfg_path = tmp_path / "cfg_missing_state.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    p_reduce = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.reduce_validate",
            "--config",
            str(cfg_path),
            "--run-id",
            "r_missing_state",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert p_reduce.returncode == 0, p_reduce.stderr
    out = json.loads(p_reduce.stdout)
    ge = dict(out["gate_evidence"])
    assert ge["state_degraded"] is True
    assert ge["state_fallback_reason"] in {
        "state_artifacts_missing_used_trace_h5",
        "state_artifacts_missing_reconstructed_from_wdot",
    }
