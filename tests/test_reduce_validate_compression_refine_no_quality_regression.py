import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml


def _run_reduce(cfg: dict, cfg_path: Path, run_id: str) -> dict:
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
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
    return json.loads(proc.stdout)


def test_reduce_validate_compression_refine_no_quality_regression(tmp_path) -> None:
    base_cfg = yaml.safe_load(Path("configs/reduce_pooling_mvp.yaml").read_text())
    base_cfg["report_dir"] = str(tmp_path / "reports")
    base_cfg.setdefault("reduction", {})["mode"] = "baseline"
    base_cfg.setdefault("evaluation", {})["surrogate_split"] = {"mode": "in_sample"}
    base_cfg.setdefault("synthetic", {})["n_reactions"] = 30

    cfg_off = copy.deepcopy(base_cfg)
    cfg_off.setdefault("evaluation", {})["compression_optimizer"] = {
        "enabled": False,
    }
    out_off = _run_reduce(cfg_off, tmp_path / "cfg_off.yaml", "compression_refine_off")

    cfg_on = copy.deepcopy(base_cfg)
    cfg_on.setdefault("evaluation", {})["compression_optimizer"] = {
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
    out_on = _run_reduce(cfg_on, tmp_path / "cfg_on.yaml", "compression_refine_on")

    sel_off = dict(out_off.get("selected_metrics") or {})
    sel_on = dict(out_on.get("selected_metrics") or {})

    mand_case_off = float(sel_off.get("pass_rate_mandatory_case", sel_off.get("pass_rate", 0.0)) or 0.0)
    mand_case_on = float(sel_on.get("pass_rate_mandatory_case", sel_on.get("pass_rate", 0.0)) or 0.0)
    mean_m_off = float(sel_off.get("mean_rel_diff_mandatory", sel_off.get("mean_rel_diff", 0.0)) or 0.0)
    mean_m_on = float(sel_on.get("mean_rel_diff_mandatory", sel_on.get("mean_rel_diff", 0.0)) or 0.0)
    mean_o_off = float(sel_off.get("mean_rel_diff_optional", sel_off.get("mean_rel_diff", 0.0)) or 0.0)
    mean_o_on = float(sel_on.get("mean_rel_diff_optional", sel_on.get("mean_rel_diff", 0.0)) or 0.0)

    assert mand_case_on + 1.0e-12 >= mand_case_off
    assert mean_m_on <= mean_m_off + 1.0e-6
    assert mean_o_on <= mean_o_off + 1.0e-6
    if bool(out_off.get("gate_passed", False)):
        assert bool(out_on.get("gate_passed", False))
