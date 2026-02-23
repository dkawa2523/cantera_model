import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_reduce_validate_pooling_bridges_to_learnckpp_path(tmp_path) -> None:
    cfg = yaml.safe_load(Path("configs/reduce_pooling_mvp.yaml").read_text())
    cfg["report_dir"] = str(tmp_path / "reports")
    cfg_path = tmp_path / "cfg_bridge.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    run_id = "pooling_bridge"
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
    ge = dict(out.get("gate_evidence") or {})
    selected = dict(ge.get("selected_stage_evidence") or {})

    assert "learnckpp_overall_candidates" in selected
    assert "learnckpp_overall_selected" in selected
    assert int(selected["learnckpp_overall_candidates"]) >= int(selected["learnckpp_overall_selected"])
    assert "pooling_hard_ban_violations" in selected
    assert int(selected["pooling_hard_ban_violations"]) >= 0

    pm = dict(out.get("pooling_metrics") or {})
    assert int(pm.get("overall_candidates", 0)) >= int(pm.get("overall_selected", 0))
