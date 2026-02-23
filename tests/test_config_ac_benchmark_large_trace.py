from pathlib import Path

import yaml


def test_config_ac_benchmark_large_trace_schema() -> None:
    path = Path("configs/ac_benchmark_large_trace.yaml")
    cfg = yaml.safe_load(path.read_text())
    surface_cfg = cfg["simulation"]["surface"]
    conditions_csv = str(cfg.get("conditions_csv", ""))
    assert cfg["simulation"]["mode"] == "surface_batch"
    assert surface_cfg["mechanism"].endswith("ac_hydrocarbon_cvd_large__gri30.yaml")
    assert surface_cfg["interface_phase"] == "ac_surf"
    assert surface_cfg["include_gas_reactions_in_trace"] is True
    assert surface_cfg["trace_wdot_policy"] == "stoich_consistent"
    assert conditions_csv.endswith("ac_hydrocarbon_cvd_large/conditions.csv")
    assert not conditions_csv.endswith("conditions_quick.csv")
    assert Path("cantera_model/benchmark_large/benchmarks/ac_hydrocarbon_cvd_large/conditions.csv").exists()
