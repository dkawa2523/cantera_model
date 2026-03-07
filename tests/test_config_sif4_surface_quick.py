from pathlib import Path

import yaml


def test_config_sif4_surface_quick_schema() -> None:
    path = Path("configs/sif4_sin3n4_cvd_quick.yaml")
    cfg = yaml.safe_load(path.read_text())
    assert cfg["simulation"]["mode"] == "surface_batch"
    assert cfg["simulation"]["surface"]["interface_phase"] == "SI3N4"
    mech = (path.parent / cfg["baseline"]["mechanism"]).resolve()
    assert mech.exists()
    assert Path("cantera_model/benchmarks/benchmarks/sif4_sin3n4_cvd/conditions.csv").exists()
