from cantera_model.eval.cantera_runner import compare_rows


def test_mandatory_tail_scope_quality_scope_ignores_invalid_units() -> None:
    baseline = []
    candidate = []
    for idx in range(10):
        case_id = f"c{idx}"
        baseline.append({"case_id": case_id, "X_last:A": 1.0, "X_last:B": 1.0})
        b_val = 20.0 if idx < 9 else 100.0
        candidate.append({"case_id": case_id, "X_last:A": 1.01, "X_last:B": b_val})
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
        "mandatory_quality_scope": "valid_only",
        "mandatory_case_pass_min": 0.75,
        "optional_metric_pass_min": 0.0,
        "max_mean_rel_diff_mandatory": 10.0,
        "max_mean_rel_diff_optional": 10.0,
        "mandatory_tail_guard_mode": "p95",
        "mandatory_tail_guard_policy": "hard",
        "mandatory_tail_rel_diff_max": 1.5,
        "mandatory_tail_min_samples": 8,
    }

    _, tail_all_units = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.2,
        mandatory_validity=mandatory_validity,
        eval_policy={"error_aggregation": {**common_error, "mandatory_tail_scope": "all_units"}},
    )
    _, tail_quality_scope = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.2,
        mandatory_validity=mandatory_validity,
        eval_policy={"error_aggregation": {**common_error, "mandatory_tail_scope": "quality_scope"}},
    )

    assert tail_all_units["mandatory_tail_scope_effective"] == "all_units"
    assert tail_quality_scope["mandatory_tail_scope_effective"] == "quality_scope"
    assert tail_all_units["mandatory_tail_guard_passed"] is False
    assert tail_quality_scope["mandatory_tail_guard_passed"] is True
    assert tail_all_units["mandatory_quality_passed"] is False
    assert tail_quality_scope["mandatory_quality_passed"] is True
    assert tail_all_units["error_fail_reason_primary"] == "mandatory_tail"
    assert tail_quality_scope["error_fail_reason_primary"] == "none"
