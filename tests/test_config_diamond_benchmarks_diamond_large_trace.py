from pathlib import Path

import yaml


def test_config_diamond_benchmarks_diamond_large_trace_schema() -> None:
    path = Path("configs/diamond_benchmarks_diamond_large_trace.yaml")
    cfg = yaml.safe_load(path.read_text())
    surface_cfg = cfg["simulation"]["surface"]
    assert cfg["simulation"]["mode"] == "surface_batch"
    assert surface_cfg["mechanism"].endswith("diamond_gri30_multisite.yaml")
    assert surface_cfg["interface_phase"] == "diamond_100_multi"
    assert surface_cfg["include_gas_reactions_in_trace"] is True
    assert surface_cfg["trace_wdot_policy"] == "stoich_consistent"
    assert Path("cantera_model/benchmarks_diamond/benchmarks/diamond_cvd_large/conditions.csv").exists()
