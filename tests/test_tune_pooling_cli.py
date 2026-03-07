import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_tune_pooling_cli_synthetic(tmp_path) -> None:
    cfg = yaml.safe_load(Path("configs/reduce_pooling_mvp.yaml").read_text())
    cfg["report_dir"] = str(tmp_path / "reports")
    cfg.setdefault("pooling", {}).setdefault("tuning", {})
    cfg["pooling"]["tuning"] = {
        "seed": 7,
        "graph_choices": ["species"],
        "backend_choices": ["numpy"],
        "temperature_range": [0.7, 0.7],
        "ratio_scale_range": [1.0, 1.0],
        "penalty_phase_range": [0.8, 0.8],
        "penalty_charge_range": [0.8, 0.8],
        "penalty_radical_range": [0.4, 0.4],
        "penalty_role_range": [0.3, 0.3],
    }

    cfg_path = tmp_path / "cfg_tune_pooling.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    out_root = tmp_path / "tuning_reports"
    run_id = "tune_pooling"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.tune_pooling",
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
    assert payload["search_backend"] in {"optuna", "random"}

    trial_dir = out_root / run_id
    assert (trial_dir / "summary.json").exists()
    assert (trial_dir / "trials.csv").exists()
    assert (trial_dir / "best_params.yaml").exists()


def test_tune_pooling_cli_apply_best_outputs_config(tmp_path) -> None:
    cfg = yaml.safe_load(Path("configs/reduce_pooling_mvp.yaml").read_text())
    cfg["report_dir"] = str(tmp_path / "reports")
    cfg.setdefault("pooling", {}).setdefault("tuning", {})
    cfg["pooling"]["tuning"] = {
        "seed": 7,
        "graph_choices": ["species"],
        "backend_choices": ["numpy"],
        "temperature_range": [0.65, 0.65],
        "ratio_scale_range": [1.0, 1.0],
        "penalty_phase_range": [0.9, 0.9],
        "penalty_charge_range": [0.7, 0.7],
        "penalty_radical_range": [0.2, 0.2],
        "penalty_role_range": [0.4, 0.4],
    }

    cfg_path = tmp_path / "cfg_tune_pooling_apply.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    out_root = tmp_path / "tuning_reports"
    run_id = "tune_pooling_apply"
    applied_path = tmp_path / "applied_pooling.yaml"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.tune_pooling",
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
    apply_best = dict(payload.get("apply_best") or {})
    assert apply_best.get("enabled") is True
    assert Path(str(apply_best.get("applied_config_path", ""))).resolve() == applied_path.resolve()
    assert int(apply_best.get("changed_count", 0)) >= 1
    changed_paths = {str(row.get("path")) for row in (apply_best.get("changed_paths") or [])}
    assert "pooling.train.temperature" in changed_paths
    assert "pooling.constraint_cfg.soft.penalty.phase" in changed_paths

    applied_cfg = yaml.safe_load(applied_path.read_text())
    assert str(applied_cfg["reduction"]["mode"]) == "pooling"
    assert str(applied_cfg["pooling"]["graph"]) == "species"
