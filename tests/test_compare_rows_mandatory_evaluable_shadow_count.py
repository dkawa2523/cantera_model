from cantera_model.eval.cantera_runner import compare_rows


def test_compare_rows_counts_shadow_evaluable_ratio_without_affecting_gate() -> None:
    baseline = [
        {"case_id": "c1", "m1": 1.0, "m2": 1.0},
        {"case_id": "c2", "m1": 1.0, "m2": 1.0},
        {"case_id": "c3", "m1": 1.0, "m2": 1.0},
        {"case_id": "c4", "m1": 1.0, "m2": 1.0},
    ]
    candidate = [
        {"case_id": "c1", "m1": 1.05, "m2": 1.0},
        {"case_id": "c2", "m1": 1.05, "m2": 1.0},
        {"case_id": "c3", "m1": 1.05, "m2": float("nan")},
        {"case_id": "c4", "m1": 1.05, "m2": float("nan")},
    ]
    mandatory_validity = {
        "mandatory_metrics": ["m1", "m2"],
        "mandatory_valid_unit_mode": "metric",
        "mandatory_metric_validity_mode": "case_pass_rate",
        "mandatory_metric_case_pass_min": 0.75,
        "mandatory_hard_mode": "min_valid_count",
        "min_valid_mandatory_count_abs": 1,
        "min_valid_mandatory_ratio": 0.0,
        "min_valid_mandatory_cap_by_total": True,
        "mandatory_gate_unit_min_evaluable_case_ratio_shadow": 0.75,
    }

    _, summary = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.20,
        mandatory_validity=mandatory_validity,
        eval_policy={"error_aggregation": {"mode": "tiered"}},
    )

    assert summary["mandatory_validity_passed"] is True
    assert summary["mandatory_gate_unit_evaluable_case_rates"]["m1"] == 1.0
    assert summary["mandatory_gate_unit_evaluable_case_rates"]["m2"] == 0.5
    assert summary["mandatory_gate_unit_valid_count_shadow_evaluable_ratio"] == 1
    assert (
        abs(summary["mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective"] - 0.75)
        < 1.0e-12
    )
