from pathlib import Path

import yaml


def test_config_diamond_benchmarks_diamond_large_quick_schema() -> None:
    path = Path("configs/diamond_benchmarks_diamond_large_quick.yaml")
    cfg = yaml.safe_load(path.read_text())
    assert cfg["simulation"]["mode"] == "surface_batch"
    assert cfg["simulation"]["surface"]["interface_phase"] == "diamond_100_multi"
    assert cfg["baseline"]["mechanism"].endswith("diamond_gri30_multisite.yaml")
    assert Path("cantera_model/benchmarks_diamond/benchmarks/diamond_cvd_large/conditions.csv").exists()
