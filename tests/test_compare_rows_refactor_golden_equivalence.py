from cantera_model.eval.cantera_runner import compare_rows


def test_compare_rows_refactor_golden_equivalence() -> None:
    baseline = [
        {"case_id": "c1", "X_last:A": 1.0, "X_max:A": 2.0, "dep_int:A": 0.5, "opt": 1.0},
        {"case_id": "c2", "X_last:A": 1.0, "X_max:A": 2.0, "dep_int:A": 0.5, "opt": 1.0},
        {"case_id": "c3", "X_last:A": 1.0, "X_max:A": 2.0, "dep_int:A": 0.5, "opt": 1.0},
    ]
    candidate = [
        {"case_id": "c1", "X_last:A": 1.05, "X_max:A": 1.8, "dep_int:A": 0.55, "opt": 0.9},
        {"case_id": "c2", "X_last:A": 1.1, "X_max:A": 2.1, "dep_int:A": 0.4, "opt": 1.4},
        {"case_id": "c3", "X_last:A": 0.8, "X_max:A": 2.4, "dep_int:A": 0.45, "opt": 0.7},
    ]

    _, summary = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.2,
        mandatory_validity={
            "mandatory_metrics": ["X_last:A", "X_max:A", "dep_int:A"],
            "mandatory_valid_unit_mode": "species_family_quorum",
            "mandatory_species_family_case_pass_min": 0.67,
            "mandatory_metric_validity_mode": "case_pass_rate",
            "mandatory_metric_case_pass_min": 0.75,
            "mandatory_hard_mode": "min_valid_count",
            "min_valid_mandatory_count_abs": 1,
            "min_valid_mandatory_ratio": 0.0,
            "min_valid_mandatory_cap_by_total": True,
        },
        eval_policy={
            "error_aggregation": {
                "mode": "tiered",
                "mandatory_case_mode": "ratio_mean",
                "mandatory_case_pass_min": 0.75,
                "optional_metric_pass_min": 0.5,
                "max_mean_rel_diff_mandatory": 0.40,
                "max_mean_rel_diff_optional": 0.60,
                "mandatory_quality_scope": "valid_only",
                "mandatory_tail_guard_mode": "p95",
                "mandatory_tail_guard_policy": "conditional_hard",
                "mandatory_tail_activation_ratio_min": 0.10,
                "mandatory_tail_rel_diff_max": 1.50,
            },
            "metric_normalization": {
                "denominator_mode": "max_abs_or_floor",
                "metric_family_abs_floor": {"default": 1.0e-12, "dep_int": 1.0e-10},
                "low_signal_case_ratio_threshold": 0.75,
                "low_signal_policy": "score_only",
            },
        },
    )

    assert abs(float(summary["pass_rate"]) - 1.0 / 3.0) < 1.0e-12
    assert abs(float(summary["mean_rel_diff"]) - 0.15833333333333335) < 1.0e-12
    assert abs(float(summary["pass_rate_mandatory_case"]) - 1.0) < 1.0e-12
    assert abs(float(summary["pass_rate_optional_metric_mean"]) - 1.0 / 3.0) < 1.0e-12
    assert abs(float(summary["mean_rel_diff_mandatory"]) - 0.12222222222222223) < 1.0e-12
    assert abs(float(summary["mean_rel_diff_optional"]) - 0.26666666666666666) < 1.0e-12
    assert summary["mandatory_error_passed"] is True
    assert summary["optional_error_passed"] is False
    assert summary["error_gate_passed"] is False
    assert summary["coverage_gate_passed"] is True
    assert summary["mandatory_validity_passed"] is True
    assert int(summary["mandatory_total_gate_unit_count"]) == 1
    assert int(summary["valid_mandatory_gate_unit_count"]) == 1
    assert abs(float(summary["mandatory_rel_diff_p95"]) - 0.2) < 1.0e-12
    assert summary["mandatory_tail_guard_triggered"] is False
    assert summary["mandatory_tail_guard_hard_applied"] is False
    assert str(summary["error_fail_reason_primary"]) == "optional_quality"
