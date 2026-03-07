import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_reduce_validate_rejects_duplicate_run_id_lock(tmp_path) -> None:
    cfg = yaml.safe_load(Path("configs/reduce_pooling_mvp.yaml").read_text())
    reports_dir = tmp_path / "reports"
    cfg["report_dir"] = str(reports_dir)
    cfg_path = tmp_path / "cfg_lock.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    run_id = "locked_run"
    run_dir = reports_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ".run.lock").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "pid": 999999,
                "started_at": "2026-01-01T00:00:00+00:00",
                "heartbeat_at": "2026-01-01T00:00:00+00:00",
                "last_progress_reason": "test_lock",
            }
        )
    )

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
    assert proc.returncode != 0
    assert "run_id lock already exists" in proc.stderr
