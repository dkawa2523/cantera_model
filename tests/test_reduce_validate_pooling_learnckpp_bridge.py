import json
import copy
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


def test_reduce_validate_pooling_bridge_light_is_faster_than_full(tmp_path) -> None:
    base_cfg = yaml.safe_load(Path("configs/reduce_pooling_mvp.yaml").read_text())
    base_cfg["report_dir"] = str(tmp_path / "reports")
    base_cfg.setdefault("synthetic", {})["n_reactions"] = 320
    base_cfg.setdefault("pooling", {}).setdefault("model", {})["backend"] = "numpy"

    full_cfg = copy.deepcopy(base_cfg)
    full_cfg.setdefault("pooling", {}).setdefault("bridge", {})
    full_cfg["pooling"]["bridge"]["enable"] = True
    full_cfg["pooling"]["bridge"]["mode"] = "full"
    full_cfg_path = tmp_path / "cfg_bridge_full.yaml"
    full_cfg_path.write_text(yaml.safe_dump(full_cfg, sort_keys=False))

    light_cfg = copy.deepcopy(base_cfg)
    light_cfg.setdefault("pooling", {}).setdefault("bridge", {})
    light_cfg["pooling"]["bridge"]["enable"] = True
    light_cfg["pooling"]["bridge"]["mode"] = "light"
    light_cfg_path = tmp_path / "cfg_bridge_light.yaml"
    light_cfg_path.write_text(yaml.safe_dump(light_cfg, sort_keys=False))

    proc_full = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.reduce_validate",
            "--config",
            str(full_cfg_path),
            "--run-id",
            "pooling_bridge_full",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_full.returncode == 0, proc_full.stderr
    out_full = json.loads(proc_full.stdout)
    full_t = float((out_full.get("selected_metrics") or {}).get("timing_bridge_s", 0.0))

    proc_light = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.reduce_validate",
            "--config",
            str(light_cfg_path),
            "--run-id",
            "pooling_bridge_light",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_light.returncode == 0, proc_light.stderr
    out_light = json.loads(proc_light.stdout)
    selected_light = dict(out_light.get("selected_metrics") or {})
    light_t = float(selected_light.get("timing_bridge_s", 0.0))

    assert str(selected_light.get("pooling_bridge_mode")) == "light"
    assert light_t <= full_t + 1.0e-9
