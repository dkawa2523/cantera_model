from cantera_model.eval.cantera_runner import compare_rows


def _rows(extreme_count: int) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    baseline = [{"case_id": f"c{i}", "m": 1.0} for i in range(20)]
    candidate = [
        {"case_id": f"c{i}", "m": (10.0 if i < extreme_count else 1.1)}
        for i in range(20)
    ]
    return baseline, candidate


def test_tail_conditional_hard_not_applied_for_small_exceed_ratio() -> None:
    baseline, candidate = _rows(1)
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
                "mandatory_outlier_ratio_max": 1.0,
                "mandatory_tail_guard_mode": "p95",
                "mandatory_tail_guard_policy": "conditional_hard",
                "mandatory_tail_rel_diff_max": 0.40,
                "mandatory_tail_min_samples": 8,
                "mandatory_tail_activation_ratio_min": 0.10,
                "mandatory_tail_exceed_ref": "tail_max",
                "optional_metric_pass_min": 0.0,
                "max_mean_rel_diff_optional": 1.0,
            }
        },
    )
    assert summary["mandatory_tail_guard_triggered"] is True
    assert summary["mandatory_tail_guard_hard_applied"] is False
    assert summary["mandatory_tail_guard_passed"] is True
    assert summary["mandatory_error_passed"] is True


def test_tail_conditional_hard_applied_for_large_exceed_ratio() -> None:
    baseline, candidate = _rows(3)
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
                "mandatory_outlier_ratio_max": 1.0,
                "mandatory_tail_guard_mode": "p95",
                "mandatory_tail_guard_policy": "conditional_hard",
                "mandatory_tail_rel_diff_max": 0.40,
                "mandatory_tail_min_samples": 8,
                "mandatory_tail_activation_ratio_min": 0.10,
                "mandatory_tail_exceed_ref": "tail_max",
                "optional_metric_pass_min": 0.0,
                "max_mean_rel_diff_optional": 1.0,
            }
        },
    )
    assert summary["mandatory_tail_guard_triggered"] is True
    assert summary["mandatory_tail_guard_hard_applied"] is True
    assert summary["mandatory_tail_guard_passed"] is False
    assert summary["mandatory_error_passed"] is False
    assert summary["error_fail_reason_primary"] == "mandatory_tail"
