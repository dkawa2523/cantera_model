import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_reduce_validate_kfold_fallback_to_in_sample(tmp_path) -> None:
    cfg = yaml.safe_load(Path("configs/reduce_surrogate_aggressive.yaml").read_text())
    cfg["report_dir"] = str(tmp_path / "reports")
    eval_cfg = dict(cfg.get("evaluation") or {})
    split_cfg = dict(eval_cfg.get("surrogate_split") or {})
    split_cfg["mode"] = "kfold"
    split_cfg["kfolds"] = 3
    eval_cfg["surrogate_split"] = split_cfg
    cfg["evaluation"] = eval_cfg

    cfg_path = tmp_path / "cfg_split.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.reduce_validate",
            "--config",
            str(cfg_path),
            "--run-id",
            "split_fallback",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    split = dict(out["surrogate_split"])
    assert split["requested_mode"] == "kfold"
    assert split["mode"] == "in_sample"
    assert split["fallback_reason"] == "insufficient_cases_for_kfold"
