from pathlib import Path

import yaml


def test_config_ac_trace_uses_full_conditions() -> None:
    path = Path("configs/ac_benchmark_large_trace.yaml")
    cfg = yaml.safe_load(path.read_text())
    conditions_csv = str(cfg.get("conditions_csv", ""))
    assert conditions_csv.endswith("ac_hydrocarbon_cvd_large/conditions.csv")
    assert not conditions_csv.endswith("conditions_quick.csv")
    assert Path("cantera_model/benchmark_large/benchmarks/ac_hydrocarbon_cvd_large/conditions.csv").exists()
