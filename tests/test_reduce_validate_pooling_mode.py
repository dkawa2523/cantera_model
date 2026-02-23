import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_reduce_validate_pooling_mode_runs(tmp_path) -> None:
    cfg = yaml.safe_load(Path("configs/reduce_pooling_mvp.yaml").read_text())
    cfg["report_dir"] = str(tmp_path / "reports")

    cfg_path = tmp_path / "cfg_pooling.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    run_id = "pooling_mode"
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
    assert out["reduction_mode"] == "pooling"
    assert "pooling_metrics" in out
    assert "pooling_artifact_path" in out
    pm = dict(out["pooling_metrics"])
    assert "overall_candidates" in pm
    assert "overall_selected" in pm
    assert "hard_ban_violations" in pm

    ge = dict(out["gate_evidence"])
    selected = dict(ge["selected_stage_evidence"])
    assert "pooling_hard_ban_violations" in selected
    assert "pooling_constraint_loss" in selected
    assert "pooling_artifact_path" in selected

    rt = dict(out.get("reduction_trace") or {})
    trend = list(rt.get("candidate_trend") or [])
    preview = dict(rt.get("cluster_preview") or {})
    assert len(trend) >= 1
    assert "selected_stage_clusters" in preview
