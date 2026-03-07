from cantera_model.cli.reduce_validate import _select_stage_physics_first


def test_pass_first_policy_prefers_gate_passed_stage() -> None:
    rows = [
        {
            "stage": "A",
            "species_before": 80,
            "reactions_before": 200,
            "species_after": 12,
            "reactions_after": 5,
            "mean_rel_diff": 0.08,
            "gate_passed": False,
            "floor_passed": True,
            "balance_gate_passed": True,
            "cluster_guard_passed": True,
            "physical_degraded": False,
            "_floors": {"min_species_after": 8, "min_reactions_after": 10},
            "_selection_max_mean_rel": 0.40,
        },
        {
            "stage": "B",
            "species_before": 80,
            "reactions_before": 200,
            "species_after": 18,
            "reactions_after": 16,
            "mean_rel_diff": 0.12,
            "gate_passed": True,
            "floor_passed": True,
            "balance_gate_passed": True,
            "cluster_guard_passed": True,
            "physical_degraded": False,
            "_floors": {"min_species_after": 8, "min_reactions_after": 10},
            "_selection_max_mean_rel": 0.40,
        },
    ]

    out = _select_stage_physics_first(rows, {"selection": {"policy": "pass_first_pareto"}})
    assert out["selected"]["stage"] == "B"


def test_pass_first_policy_applies_tie_breakers_for_compression() -> None:
    rows = [
        {
            "stage": "A",
            "species_before": 100,
            "reactions_before": 200,
            "species_after": 30,
            "reactions_after": 60,
            "mean_rel_diff": 0.20,
            "gate_passed": True,
            "floor_passed": True,
            "balance_gate_passed": True,
            "cluster_guard_passed": True,
            "physical_degraded": False,
            "_floors": {"min_species_after": 8, "min_reactions_after": 10},
            "_selection_max_mean_rel": 0.40,
        },
        {
            "stage": "B",
            "species_before": 100,
            "reactions_before": 200,
            "species_after": 30,
            "reactions_after": 40,
            "mean_rel_diff": 0.20,
            "gate_passed": True,
            "floor_passed": True,
            "balance_gate_passed": True,
            "cluster_guard_passed": True,
            "physical_degraded": False,
            "_floors": {"min_species_after": 8, "min_reactions_after": 10},
            "_selection_max_mean_rel": 0.40,
        },
    ]

    out = _select_stage_physics_first(
        rows,
        {
            "selection": {
                "policy": "pass_first_pareto",
                "weights": {
                    "err": 0.0,
                    "reaction_comp": 0.0,
                    "species_comp": 0.0,
                    "floor_margin": 0.0,
                    "balance_margin": 0.0,
                    "coverage_margin": 0.0,
                    "cluster_size_margin": 0.0,
                },
                "tie_breakers": ["reaction_reduction", "species_reduction", "mean_rel_diff"],
            }
        },
    )
    assert out["selected"]["stage"] == "B"
