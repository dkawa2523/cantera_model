import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_reduce_validate_summary_contains_balance_margin_detail(tmp_path: Path) -> None:
    cfg = yaml.safe_load(Path("configs/reduce_surrogate_aggressive.yaml").read_text())
    cfg["report_dir"] = str(tmp_path / "reports")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    run_id = "balance_margin_detail_run"
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
    gate_evidence = dict(out.get("gate_evidence") or {})
    detail = dict(gate_evidence.get("balance_margin_detail") or {})
    assert detail.get("selected_stage")
    assert isinstance(detail.get("selected"), dict)
    assert isinstance(detail.get("per_stage"), dict)

    selected = dict(detail.get("selected") or {})
    ratio_detail = dict(selected.get("reaction_species_ratio") or {})
    assert "min_margin" in ratio_detail
    assert "max_margin" in ratio_detail
