from cantera_model.eval.cantera_runner import compare_rows


def test_compare_rows_mandatory_winsorized_mean_is_lower_than_raw() -> None:
    baseline = [{"case_id": f"c{i}", "m": 1.0} for i in range(10)]
    candidate = [{"case_id": f"c{i}", "m": (10.0 if i == 0 else 1.1)} for i in range(10)]

    _, summary = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=10.0,
        mandatory_validity={"mandatory_metrics": ["m"]},
        eval_policy={
            "error_aggregation": {
                "mode": "tiered",
                "mandatory_case_pass_min": 0.0,
                "max_mean_rel_diff_mandatory": 0.40,
                "mandatory_mean_mode": "winsorized",
                "mandatory_winsor_cap_multiplier": 3.0,
                "mandatory_outlier_multiplier": 5.0,
                "mandatory_outlier_ratio_max": 0.20,
            }
        },
    )

    raw = float(summary["mean_rel_diff_mandatory_raw"])
    winsor = float(summary["mean_rel_diff_mandatory_winsorized"])
    assert winsor < raw
    assert float(summary["mean_rel_diff_mandatory"]) == winsor
    assert abs(float(summary["mandatory_rel_outlier_ratio"]) - 0.1) < 1.0e-12


def test_compare_rows_mandatory_outlier_ratio_gate_can_fail() -> None:
    baseline = [{"case_id": f"c{i}", "m": 1.0} for i in range(10)]
    candidate = [{"case_id": f"c{i}", "m": (10.0 if i < 3 else 1.1)} for i in range(10)]

    _, summary = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=10.0,
        mandatory_validity={"mandatory_metrics": ["m"]},
        eval_policy={
            "error_aggregation": {
                "mode": "tiered",
                "mandatory_case_pass_min": 0.0,
                "max_mean_rel_diff_mandatory": 0.40,
                "mandatory_mean_mode": "winsorized",
                "mandatory_winsor_cap_multiplier": 3.0,
                "mandatory_outlier_multiplier": 5.0,
                "mandatory_outlier_ratio_max": 0.20,
            }
        },
    )

    assert float(summary["mandatory_rel_outlier_ratio"]) == 0.3
    assert float(summary["mandatory_rel_outlier_ratio_max_effective"]) == 0.2
    assert summary["mandatory_error_passed"] is False
