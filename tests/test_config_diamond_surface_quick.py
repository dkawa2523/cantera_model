from pathlib import Path

import yaml


def test_config_diamond_surface_quick_schema() -> None:
    path = Path("configs/diamond_cvd_quick.yaml")
    cfg = yaml.safe_load(path.read_text())
    assert cfg["simulation"]["mode"] == "surface_batch"
    assert cfg["simulation"]["surface"]["interface_phase"] == "diamond_100"
    assert cfg["baseline"]["mechanism"] == "diamond.yaml"
    assert Path("cantera_model/benchmarks/benchmarks/diamond_cvd/conditions.csv").exists()
