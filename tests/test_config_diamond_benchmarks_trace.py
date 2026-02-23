from pathlib import Path

import yaml


def test_config_diamond_benchmarks_trace_schema() -> None:
    path = Path("configs/diamond_benchmarks_diamond_trace.yaml")
    cfg = yaml.safe_load(path.read_text())
    assert cfg["simulation"]["mode"] == "surface_batch"
    assert cfg["simulation"]["surface"]["mechanism"] == "diamond.yaml"
    assert cfg["simulation"]["surface"]["interface_phase"] == "diamond_100"
    assert Path("cantera_model/benchmarks_diamond/benchmarks/diamond_cvd/conditions.csv").exists()
