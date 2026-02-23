from cantera_model.cli.reduce_validate import _evaluate_balance_gate


def test_evaluate_balance_gate_detects_low_ratio() -> None:
    bands = {
        "enabled": True,
        "min_reaction_species_ratio": 0.4,
        "max_reaction_species_ratio": 4.0,
        "min_active_species_coverage": 0.8,
        "min_nu_rank_ratio": 0.35,
    }
    metrics = {
        "reaction_species_ratio": 0.2,
        "active_species_coverage": 0.9,
        "nu_rank_ratio": 0.9,
    }
    out = _evaluate_balance_gate(metrics, bands)
    assert out["passed"] is False
    assert "min_reaction_species_ratio" in out["violations"]
