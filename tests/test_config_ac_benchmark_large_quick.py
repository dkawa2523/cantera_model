from pathlib import Path

import yaml


def test_config_ac_benchmark_large_quick_schema() -> None:
    path = Path("configs/ac_benchmark_large_quick.yaml")
    cfg = yaml.safe_load(path.read_text())
    assert cfg["simulation"]["mode"] == "surface_batch"
    assert cfg["simulation"]["surface"]["interface_phase"] == "ac_surf"
    assert cfg["baseline"]["mechanism"].endswith("ac_hydrocarbon_cvd_large__gri30.yaml")
    assert Path("cantera_model/benchmark_large/benchmarks/ac_hydrocarbon_cvd_large/conditions_quick.csv").exists()
