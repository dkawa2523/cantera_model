import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_reduce_validate_learnckpp_fallback_to_baseline(tmp_path) -> None:
    cfg = yaml.safe_load(Path("configs/reduce_learnckpp_mvp.yaml").read_text())
    cfg["report_dir"] = str(tmp_path / "reports")
    cfg.setdefault("learnckpp", {})
    cfg["learnckpp"]["fallback_to_baseline_on_error"] = True
    cfg["learnckpp"].setdefault("candidate", {})["min_flux_quantile"] = 0.0
    cfg["learnckpp"].setdefault("select", {})["method"] = "invalid_method"

    cfg_path = tmp_path / "cfg_fallback.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    run_id = "learnckpp_fallback"
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
    assert out["failure_reason"] == "learnckpp_stage_failed_fallback_baseline"
    fb = dict(out["learnckpp_fallback"])
    assert fb["enabled"] is True
    assert fb["triggered"] is True
    assert len(fb["reasons"]) >= 1

    selected = dict(out["selected_metrics"])
    assert selected["reduction_mode"] == "baseline_fallback"
    assert str(selected["prune_status"]).startswith("learnckpp_failed_baseline:")
