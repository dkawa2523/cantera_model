from cantera_model.cli.reduce_validate import _select_stage_pass_first


def _base_row(stage: str, raw_drift: float) -> dict[str, object]:
    return {
        "stage": stage,
        "species_before": 100,
        "reactions_before": 200,
        "species_after": 40,
        "reactions_after": 90,
        "mean_rel_diff": 0.20,
        "gate_passed": True,
        "floor_passed": True,
        "balance_gate_passed": True,
        "cluster_guard_passed": True,
        "physical_gate_passed": True,
        "physical_degraded": False,
        "balance_margin": 0.05,
        "_floors": {"min_species_after": 8, "min_reactions_after": 10},
        "_selection_max_mean_rel": 0.40,
        "_selection_use_raw_drift": True,
        "_selection_raw_drift_cap": 2.0,
        "metric_drift_raw": raw_drift,
        "metric_drift_effective": 1.30,
    }


def test_selection_uses_raw_drift_quality_as_tiebreak() -> None:
    rows = [
        _base_row("A", raw_drift=1.05),
        _base_row("B", raw_drift=1.90),
    ]

    out = _select_stage_pass_first(rows, {"selection": {"policy": "pass_first_pareto"}})
    selected = out["selected"]
    assert selected["stage"] == "A"
    assert float(rows[0]["selection_quality_score_raw_drift"]) > float(
        rows[1]["selection_quality_score_raw_drift"]
    )
