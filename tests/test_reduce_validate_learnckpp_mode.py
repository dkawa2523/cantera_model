import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_reduce_validate_learnckpp_mode_runs(tmp_path) -> None:
    cfg = yaml.safe_load(Path("configs/reduce_learnckpp_mvp.yaml").read_text())
    cfg["report_dir"] = str(tmp_path / "reports")

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    run_id = "learnckpp_mode"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.reduce_validate",
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

    out = json.loads(proc.stdout)
    assert out["reduction_mode"] == "learnckpp"
    assert "learnckpp_metrics" in out
    assert "target_keep_ratio" in out["learnckpp_metrics"]
    assert "adaptive_keep_ratio" in out["learnckpp_metrics"]
    ge = dict(out["gate_evidence"])
    selected_evidence = dict(ge["selected_stage_evidence"])
    assert "learnckpp_overall_candidates" in selected_evidence
    assert "learnckpp_overall_selected" in selected_evidence
    assert "learnckpp_target_keep_ratio" in selected_evidence
    policy = dict(selected_evidence.get("learnckpp_keep_ratio_policy") or {})
    assert "auto_tune" in policy

    metrics_path = tmp_path / "reports" / run_id / "metrics.csv"
    with metrics_path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) >= 3
    for row in rows:
        assert int(row["overall_candidates"]) >= int(row["overall_selected"])
        assert float(row["learnckpp_target_keep_ratio"]) > 0.0
