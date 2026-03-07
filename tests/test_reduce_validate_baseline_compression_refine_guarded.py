import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_reduce_validate_baseline_compression_refine_guarded(tmp_path) -> None:
    cfg = yaml.safe_load(Path("configs/reduce_pooling_mvp.yaml").read_text())
    cfg["report_dir"] = str(tmp_path / "reports")
    cfg.setdefault("reduction", {})["mode"] = "baseline"
    cfg.setdefault("evaluation", {})["surrogate_split"] = {"mode": "in_sample"}
    cfg["evaluation"]["compression_optimizer"] = {
        "enabled": True,
        "mode": "deterministic_grid",
        "per_stage_extra_trials": 2,
        "reaction_priority": True,
        "max_allowed_mandatory_mean_delta": 1.0e-6,
        "max_allowed_optional_mean_delta": 1.0e-6,
        "max_allowed_mandatory_pass_rate_drop": 0.0,
        "require_gate_passed": True,
        "require_structure_passed": True,
    }
    cfg.setdefault("synthetic", {})["n_reactions"] = 28

    cfg_path = tmp_path / "cfg_baseline_refine.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.reduce_validate",
            "--config",
            str(cfg_path),
            "--run-id",
            "baseline_refine_guarded",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    out = json.loads(proc.stdout)
    selected = dict(out.get("selected_metrics") or {})
    selected_evidence = dict(((out.get("gate_evidence") or {}).get("selected_stage_evidence") or {}))

    assert str(selected.get("compression_refine_mode_effective")) == "baseline_grid"
    assert int(selected.get("compression_refine_trials", 0)) == 2
    assert bool(selected.get("compression_refine_guard_passed", False)) is True
    assert "compression_refine_applied" in selected
    assert int(selected.get("compression_refine_reaction_delta", 0)) >= 0
    assert int(selected.get("compression_refine_species_delta", 0)) >= 0
    assert "compression_refine_applied" in selected_evidence
    assert "compression_refine_mode_effective" in selected_evidence
