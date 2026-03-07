from cantera_model.eval.cantera_runner import compare_rows


def test_mandatory_quality_scope_valid_only_excludes_invalid_units_from_hard_gate() -> None:
    baseline = [
        {"case_id": "c1", "X_last:A": 1.0, "X_last:B": 1.0},
        {"case_id": "c2", "X_last:A": 1.0, "X_last:B": 1.0},
    ]
    candidate = [
        {"case_id": "c1", "X_last:A": 1.0, "X_last:B": 1.5},
        {"case_id": "c2", "X_last:A": 1.0, "X_last:B": 1.5},
    ]
    mandatory_validity = {
        "mandatory_metrics": ["X_last:A", "X_last:B"],
        "mandatory_valid_unit_mode": "metric",
        "mandatory_metric_validity_mode": "case_pass_rate",
        "mandatory_metric_case_pass_min": 0.75,
        "mandatory_hard_mode": "min_valid_count",
        "min_valid_mandatory_count_abs": 1,
        "min_valid_mandatory_ratio": 0.0,
        "min_valid_mandatory_cap_by_total": True,
    }
    common_error = {
        "mode": "tiered",
        "mandatory_case_pass_min": 0.75,
        "optional_metric_pass_min": 0.0,
        "max_mean_rel_diff_mandatory": 10.0,
        "max_mean_rel_diff_optional": 10.0,
        "mandatory_tail_guard_mode": "none",
    }

    _, all_units = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.2,
        mandatory_validity=mandatory_validity,
        eval_policy={"error_aggregation": {**common_error, "mandatory_quality_scope": "all_units"}},
    )
    _, valid_only = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.2,
        mandatory_validity=mandatory_validity,
        eval_policy={"error_aggregation": {**common_error, "mandatory_quality_scope": "valid_only"}},
    )

    assert all_units["mandatory_quality_scope_effective"] == "all_units"
    assert valid_only["mandatory_quality_scope_effective"] == "valid_only"
    assert float(valid_only["pass_rate_mandatory_case"]) >= float(all_units["pass_rate_mandatory_case"])
    assert float(valid_only["pass_rate_mandatory_case"]) == 1.0
    assert float(valid_only["pass_rate_mandatory_case_all_units"]) == float(all_units["pass_rate_mandatory_case"])
    assert all_units["mandatory_quality_passed"] is False
    assert valid_only["mandatory_quality_passed"] is True
    assert all_units["error_fail_reason_primary"] == "mandatory_case_rate"
    assert valid_only["error_fail_reason_primary"] == "none"
