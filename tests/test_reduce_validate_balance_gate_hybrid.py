from cantera_model.cli.reduce_validate import _evaluate_balance_gate


def test_evaluate_balance_gate_hybrid_checks_weighted_coverage() -> None:
    bands = {
        "enabled": True,
        "balance_mode": "hybrid",
        "min_reaction_species_ratio": 0.3,
        "max_reaction_species_ratio": 6.0,
        "min_active_species_coverage": 0.2,
        "min_weighted_active_species_coverage": 0.8,
        "min_essential_species_coverage": 0.7,
        "min_nu_rank_ratio": 0.3,
    }
    metrics = {
        "reaction_species_ratio": 0.6,
        "active_species_coverage": 0.5,
        "weighted_active_species_coverage": 0.72,
        "essential_species_coverage": 0.95,
        "nu_rank_ratio": 0.9,
    }
    out = _evaluate_balance_gate(metrics, bands)
    assert out["passed"] is False
    assert "min_weighted_active_species_coverage" in out["violations"]


def test_evaluate_balance_gate_hybrid_pass() -> None:
    bands = {
        "enabled": True,
        "balance_mode": "hybrid",
        "min_reaction_species_ratio": 0.3,
        "max_reaction_species_ratio": 6.0,
        "min_active_species_coverage": 0.3,
        "min_weighted_active_species_coverage": 0.8,
        "min_essential_species_coverage": 0.8,
        "min_nu_rank_ratio": 0.3,
    }
    metrics = {
        "reaction_species_ratio": 1.2,
        "active_species_coverage": 0.6,
        "weighted_active_species_coverage": 0.9,
        "essential_species_coverage": 0.85,
        "nu_rank_ratio": 0.7,
    }
    out = _evaluate_balance_gate(metrics, bands)
    assert out["passed"] is True


def test_evaluate_balance_gate_hybrid_relaxes_essential_when_weighted_is_good() -> None:
    bands = {
        "enabled": True,
        "balance_mode": "hybrid",
        "min_reaction_species_ratio": 0.3,
        "max_reaction_species_ratio": 6.0,
        "min_active_species_coverage": 0.35,
        "min_weighted_active_species_coverage": 0.78,
        "min_essential_species_coverage": 0.8,
        "essential_relax_when_weighted_passed": True,
        "min_essential_species_coverage_relaxed": 0.6,
        "min_nu_rank_ratio": 0.3,
    }
    metrics = {
        "reaction_species_ratio": 0.6,
        "active_species_coverage": 0.40,
        "weighted_active_species_coverage": 0.90,
        "essential_species_coverage": 0.62,
        "nu_rank_ratio": 0.8,
    }
    out = _evaluate_balance_gate(metrics, bands)
    assert out["passed"] is True
