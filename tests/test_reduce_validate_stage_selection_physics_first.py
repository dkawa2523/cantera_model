from cantera_model.cli.reduce_validate import _select_stage_physics_first


def test_stage_selection_prefers_floor_passing_candidate() -> None:
    rows = [
        {
            "stage": "A",
            "species_before": 50,
            "reactions_before": 100,
            "species_after": 3,
            "reactions_after": 1,
            "mean_rel_diff": 0.05,
            "gate_passed": False,
            "floor_passed": False,
            "physical_degraded": False,
            "_floors": {"min_species_after": 8, "min_reactions_after": 10},
            "_selection_max_mean_rel": 0.40,
        },
        {
            "stage": "B",
            "species_before": 50,
            "reactions_before": 100,
            "species_after": 10,
            "reactions_after": 14,
            "mean_rel_diff": 0.12,
            "gate_passed": True,
            "floor_passed": True,
            "physical_degraded": False,
            "_floors": {"min_species_after": 8, "min_reactions_after": 10},
            "_selection_max_mean_rel": 0.40,
        },
    ]
    result = _select_stage_physics_first(rows, {"selection": {"policy": "physics_first_pareto"}})
    selected = result["selected"]
    assert selected["stage"] == "B"
    assert float(selected["selection_score"]) > float(rows[0].get("selection_score", -999.0))
