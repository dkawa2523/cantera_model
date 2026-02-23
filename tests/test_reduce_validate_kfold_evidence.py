import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_reduce_validate_adaptive_kfold_evidence_in_summary(tmp_path: Path) -> None:
    cfg = yaml.safe_load(Path("configs/reduce_surrogate_aggressive.yaml").read_text())
    cfg["report_dir"] = str(tmp_path / "reports")
    cfg["qoi"] = {
        "species_last": ["CO2", "CO"],
        "species_max": ["OH"],
        "species_integral": ["CO2"],
        "deposition_integral": ["CO"],
    }
    eval_cfg = dict(cfg.get("evaluation") or {})
    split_cfg = dict(eval_cfg.get("surrogate_split") or {})
    split_cfg["mode"] = "adaptive_kfold"
    split_cfg["kfold_policy"] = {
        "min_cases_for_kfold": 4,
        "default_k": 2,
        "k_by_case_count": {6: 3, 8: 3},
    }
    eval_cfg["surrogate_split"] = split_cfg
    cfg["evaluation"] = eval_cfg

    cfg_path = tmp_path / "cfg_adaptive.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.reduce_validate",
            "--config",
            str(cfg_path),
            "--run-id",
            "adaptive_kfold_evidence",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)

    split = dict(payload["surrogate_split"])
    assert split["requested_mode"] == "adaptive_kfold"
    assert "case_count" in split
    assert "effective_kfolds" in split
    assert "fold_sizes" in split

    gate_evidence = dict(payload["gate_evidence"])
    kfold_evidence = dict(gate_evidence["kfold_fold_metrics"])
    assert "selected" in kfold_evidence
    assert "per_stage" in kfold_evidence
    assert isinstance(kfold_evidence["selected"], list)

    selected = dict(payload["selected_metrics"])
    assert "split_mode" in selected
    assert "qoi_metrics_count" in selected
    assert "integral_qoi_count" in selected
