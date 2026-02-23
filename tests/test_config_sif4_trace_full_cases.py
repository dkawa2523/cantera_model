from pathlib import Path

import yaml


def test_config_sif4_trace_uses_full_conditions() -> None:
    path = Path("configs/sif4_benchmark_sin3n4_large_trace.yaml")
    cfg = yaml.safe_load(path.read_text())
    conditions_csv = str(cfg.get("conditions_csv", ""))
    assert conditions_csv.endswith("sif4_sin3n4_cvd_large/conditions.csv")
    assert not conditions_csv.endswith("conditions_quick.csv")
    assert Path("cantera_model/benchmark_sif4_sin3n4_cvd/benchmarks/sif4_sin3n4_cvd_large/conditions.csv").exists()
