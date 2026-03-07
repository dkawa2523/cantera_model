from pathlib import Path

import yaml


def test_sif4_large_configs_enable_dynamic_balance() -> None:
    for name in [
        "configs/reduce_sif4_benchmark_sin3n4_large_baseline.yaml",
        "configs/reduce_sif4_benchmark_sin3n4_large_learnckpp.yaml",
        "configs/reduce_sif4_benchmark_sin3n4_large_pooling.yaml",
    ]:
        cfg = yaml.safe_load(Path(name).read_text())
        assert cfg["selection"]["policy"] == "pass_first_pareto"
        assert cfg["selection"]["tie_breakers"] == ["reaction_reduction", "species_reduction", "mean_rel_diff"]
        dynamic = cfg["balance_constraints"]["dynamic"]
        assert dynamic["enabled"] is True
