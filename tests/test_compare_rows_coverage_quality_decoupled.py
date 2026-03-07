from cantera_model.eval.cantera_runner import compare_rows


def test_error_gate_can_pass_when_coverage_fails_if_decoupled() -> None:
    baseline = [
        {"case_id": "c1", "m1": 1.0, "m2": 1.0},
        {"case_id": "c2", "m1": 1.0, "m2": 1.0},
    ]
    candidate = [
        {"case_id": "c1", "m1": float("nan"), "m2": 1.0},
        {"case_id": "c2", "m1": float("nan"), "m2": 1.0},
    ]

    _, summary = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.20,
        mandatory_validity={
            "mandatory_metrics": ["m1", "m2"],
            "mandatory_metric_validity_mode": "all_cases",
            "mandatory_hard_mode": "min_valid_count",
            "min_valid_mandatory_count_abs": 2,
            "min_valid_mandatory_ratio": 1.0,
            "min_valid_mandatory_cap_by_total": True,
            "mandatory_valid_unit_mode": "metric",
        },
        eval_policy={
            "error_aggregation": {
                "mode": "tiered",
                "mandatory_error_include_validity": False,
                "mandatory_case_pass_min": 0.50,
                "max_mean_rel_diff_mandatory": 0.40,
                "mandatory_mean_mode": "winsorized",
                "mandatory_winsor_cap_multiplier": 3.0,
                "mandatory_outlier_multiplier": 5.0,
                "mandatory_outlier_ratio_max": 0.20,
                "mandatory_tail_guard_mode": "none",
                "optional_metric_pass_min": 0.0,
                "max_mean_rel_diff_optional": 1.0,
            }
        },
    )

    assert summary["mandatory_validity_passed"] is False
    assert summary["coverage_gate_passed"] is False
    assert summary["mandatory_quality_passed"] is True
    assert summary["mandatory_error_passed"] is True
    assert summary["error_gate_passed"] is True
    assert summary["error_fail_reason_primary"] == "none"
