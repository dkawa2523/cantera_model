import csv
import subprocess
import sys
from pathlib import Path

import yaml


def test_reduce_validate_stage_trend(tmp_path) -> None:
    src_cfg = Path("configs/reduce_surrogate_aggressive.yaml")
    cfg = yaml.safe_load(src_cfg.read_text())
    cfg["report_dir"] = str(tmp_path / "reports")

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.reduce_validate",
            "--config",
            str(cfg_path),
            "--run-id",
            "t0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    metrics_path = tmp_path / "reports" / "t0" / "metrics.csv"
    with metrics_path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))

    species_after = [int(r["species_after"]) for r in rows]
    reactions_after = [int(r["reactions_after"]) for r in rows]
    hard_ban = [int(r["hard_ban_violations"]) for r in rows]

    assert species_after[0] >= species_after[1] >= species_after[2]
    assert reactions_after[0] >= reactions_after[1] >= reactions_after[2]
    assert all(v == 0 for v in hard_ban)
