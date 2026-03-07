import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_tune_keep_ratio_cli_synthetic(tmp_path) -> None:
    cfg = yaml.safe_load(Path("configs/reduce_learnckpp_mvp.yaml").read_text())
    cfg["report_dir"] = str(tmp_path / "reports")
    cfg.setdefault("learnckpp", {}).setdefault("adaptive_keep_ratio", {}).setdefault("calibration", {})
    cfg["learnckpp"]["adaptive_keep_ratio"]["calibration"] = {
        "safety_weight_values": [2.0],
        "risk_keep_boost_values": [0.35],
        "source_risk_trace_values": [0.40],
        "source_risk_network_values": [0.40],
    }

    cfg_path = tmp_path / "cfg_tune.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    out_root = tmp_path / "tuning_reports"
    run_id = "tune_synth"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.tune_keep_ratio",
            "--config",
            str(cfg_path),
            "--run-id",
            run_id,
            "--output-root",
            str(out_root),
            "--max-trials",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["trials"] == 1
    assert payload["modes"] == ["synthetic"]

    trial_dir = out_root / run_id
    assert (trial_dir / "summary.json").exists()
    assert (trial_dir / "trials.csv").exists()
    assert (trial_dir / "best_params.yaml").exists()


def test_tune_keep_ratio_cli_apply_best_outputs_config(tmp_path) -> None:
    cfg = yaml.safe_load(Path("configs/reduce_learnckpp_mvp.yaml").read_text())
    cfg["report_dir"] = str(tmp_path / "reports")
    cfg.setdefault("learnckpp", {}).setdefault("adaptive_keep_ratio", {}).setdefault("calibration", {})
    cfg["learnckpp"]["adaptive_keep_ratio"]["calibration"] = {
        "safety_weight_values": [1.75],
        "risk_keep_boost_values": [0.31],
        "source_risk_trace_values": [0.42],
        "source_risk_network_values": [0.43],
    }

    cfg_path = tmp_path / "cfg_tune_apply.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    out_root = tmp_path / "tuning_reports"
    run_id = "tune_apply"
    applied_path = tmp_path / "applied.yaml"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.tune_keep_ratio",
            "--config",
            str(cfg_path),
            "--run-id",
            run_id,
            "--output-root",
            str(out_root),
            "--max-trials",
            "1",
            "--apply-best",
            "--applied-config-out",
            str(applied_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert "apply_best" in payload
    apply_best = dict(payload["apply_best"])
    assert apply_best["enabled"] is True
    assert Path(apply_best["applied_config_path"]).resolve() == applied_path.resolve()
    assert applied_path.exists()
    assert int(apply_best["changed_count"]) >= 1
    changed_paths = {row["path"] for row in (apply_best.get("changed_paths") or [])}
    assert "learnckpp.adaptive_keep_ratio.auto_tune.safety_weight" in changed_paths
    assert "learnckpp.adaptive_keep_ratio.auto_tune.risk_keep_boost" in changed_paths

    applied_cfg = yaml.safe_load(applied_path.read_text())
    auto = (
        applied_cfg.get("learnckpp", {})
        .get("adaptive_keep_ratio", {})
        .get("auto_tune", {})
    )
    assert float(auto["safety_weight"]) == 1.75
    assert float(auto["risk_keep_boost"]) == 0.31
    src = dict(auto.get("source_risk") or {})
    assert float(src["trace_h5"]) == 0.42
    assert float(src["network_dir"]) == 0.43


def test_tune_keep_ratio_cli_apply_inplace_writes_backup(tmp_path) -> None:
    cfg = yaml.safe_load(Path("configs/reduce_learnckpp_mvp.yaml").read_text())
    cfg["report_dir"] = str(tmp_path / "reports")
    cfg.setdefault("learnckpp", {}).setdefault("adaptive_keep_ratio", {}).setdefault("calibration", {})
    cfg["learnckpp"]["adaptive_keep_ratio"]["calibration"] = {
        "safety_weight_values": [1.61],
        "risk_keep_boost_values": [0.29],
        "source_risk_trace_values": [0.33],
        "source_risk_network_values": [0.34],
    }

    cfg_path = tmp_path / "cfg_inplace.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    out_root = tmp_path / "tuning_reports"
    run_id = "tune_inplace"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.tune_keep_ratio",
            "--config",
            str(cfg_path),
            "--run-id",
            run_id,
            "--output-root",
            str(out_root),
            "--max-trials",
            "1",
            "--apply-best",
            "--apply-inplace",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    apply_best = dict(payload.get("apply_best") or {})
    assert apply_best.get("inplace") is True
    backup_path = Path(str(apply_best.get("backup_path", "")))
    assert backup_path.exists()
    assert Path(str(apply_best.get("inplace_target", ""))).resolve() == cfg_path.resolve()

    updated_cfg = yaml.safe_load(cfg_path.read_text())
    auto = updated_cfg["learnckpp"]["adaptive_keep_ratio"]["auto_tune"]
    assert float(auto["safety_weight"]) == 1.61
    assert float(auto["risk_keep_boost"]) == 0.29
