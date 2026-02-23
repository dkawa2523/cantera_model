import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_reduce_validate_summary_contains_gate_evidence(tmp_path) -> None:
    cfg = yaml.safe_load(Path("configs/reduce_surrogate_aggressive.yaml").read_text())
    cfg["report_dir"] = str(tmp_path / "reports")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    run_id = "gate_evidence_run"
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
    assert "gate_evidence" in out
    ge = dict(out["gate_evidence"])
    assert "selected_stage" in ge
    assert "selected_stage_evidence" in ge
    assert "per_stage" in ge
    assert "state_source" in ge
