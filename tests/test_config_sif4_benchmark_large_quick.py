from pathlib import Path

import yaml


def test_config_sif4_benchmark_large_quick_schema() -> None:
    path = Path("configs/sif4_benchmark_sin3n4_large_quick.yaml")
    cfg = yaml.safe_load(path.read_text())
    assert cfg["simulation"]["mode"] == "surface_batch"
    assert cfg["simulation"]["surface"]["interface_phase"] == "SI3N4"
    assert cfg["baseline"]["mechanism"].endswith("SiF4_NH3_mec_large__gri30__multisite3.yaml")
    assert Path(
        "cantera_model/benchmark_sif4_sin3n4_cvd/benchmarks/sif4_sin3n4_cvd_large/conditions_quick.csv"
    ).exists()
