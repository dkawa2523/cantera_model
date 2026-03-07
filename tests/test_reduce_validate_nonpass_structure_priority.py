from cantera_model.cli.reduce_validate import _select_stage_physics_first


def test_nonpass_selection_prefers_smaller_structure_deficit() -> None:
    rows = [
        {
            "stage": "A",
            "species_before": 100,
            "reactions_before": 200,
            "species_after": 30,
            "reactions_after": 80,
            "mean_rel_diff": 0.20,
            "gate_passed": False,
            "floor_passed": True,
            "balance_gate_passed": True,
            "cluster_guard_passed": True,
            "physical_gate_passed": True,
            "physical_degraded": False,
            "balance_margin": -0.05,
            "_floors": {"min_species_after": 8, "min_reactions_after": 10},
            "_selection_max_mean_rel": 0.40,
        },
        {
            "stage": "B",
            "species_before": 100,
            "reactions_before": 200,
            "species_after": 12,
            "reactions_after": 40,
            "mean_rel_diff": 0.20,
            "gate_passed": False,
            "floor_passed": True,
            "balance_gate_passed": True,
            "cluster_guard_passed": True,
            "physical_gate_passed": True,
            "physical_degraded": False,
            "balance_margin": -0.40,
            "_floors": {"min_species_after": 8, "min_reactions_after": 10},
            "_selection_max_mean_rel": 0.40,
        },
    ]

    out = _select_stage_physics_first(
        rows,
        {"selection": {"policy": "physics_first_pareto", "nonpass_priority": "structure_then_score"}},
    )
    assert out["selected"]["stage"] == "A"
    assert out["selected"]["selection_pool_kind"] == "floor"
    assert float(out["selected"]["structure_deficit_score"]) < float(rows[1]["structure_deficit_score"])
