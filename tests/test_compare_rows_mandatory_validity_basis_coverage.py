from cantera_model.eval.cantera_runner import compare_rows


def test_compare_rows_mandatory_validity_basis_coverage_evaluable() -> None:
    baseline = [
        {"case_id": "c1", "m1": 1.0, "m2": 1.0},
        {"case_id": "c2", "m1": 1.0, "m2": 1.0},
    ]
    candidate = [
        {"case_id": "c1", "m1": 1.5, "m2": 1.0},
        {"case_id": "c2", "m1": 1.5, "m2": 1.0},
    ]

    _, summary = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.20,
        mandatory_validity={
            "mandatory_metrics": ["m1", "m2"],
            "mandatory_valid_unit_mode": "metric",
            "mandatory_metric_validity_mode": "case_pass_rate",
            "mandatory_metric_case_pass_min": 0.75,
            "mandatory_validity_basis": "coverage_evaluable",
            "mandatory_hard_mode": "min_valid_count",
            "min_valid_mandatory_count_abs": 2,
            "min_valid_mandatory_ratio": 1.0,
            "min_valid_mandatory_cap_by_total": True,
        },
        eval_policy={"error_aggregation": {"mode": "tiered"}},
    )

    assert summary["mandatory_validity_basis_effective"] == "coverage_evaluable"
    assert summary["valid_mandatory_gate_unit_count_case_rate"] == 1
    assert summary["valid_mandatory_gate_unit_count_coverage"] == 2
    assert summary["valid_mandatory_gate_unit_count"] == 2
    assert summary["mandatory_validity_passed"] is True
    assert abs(float(summary["mandatory_gate_unit_evaluable_case_rates"]["m1"]) - 1.0) < 1.0e-12
    assert abs(float(summary["mandatory_gate_unit_evaluable_case_rates"]["m2"]) - 1.0) < 1.0e-12
