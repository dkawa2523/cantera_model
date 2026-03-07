from cantera_model.eval.cantera_runner import compare_rows


def _rows() -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    baseline = [{"case_id": f"c{i}", "m": 1.0} for i in range(20)]
    candidate = [{"case_id": f"c{i}", "m": (10.0 if i == 0 else 1.1)} for i in range(20)]
    return baseline, candidate


def test_mandatory_tail_guard_p95_blocks_heavy_tail_even_when_winsorized_passes() -> None:
    baseline, candidate = _rows()
    common_policy = {
        "mode": "tiered",
        "mandatory_case_pass_min": 0.0,
        "max_mean_rel_diff_mandatory": 0.40,
        "mandatory_mean_mode": "winsorized",
        "mandatory_winsor_cap_multiplier": 3.0,
        "mandatory_outlier_multiplier": 5.0,
        "mandatory_outlier_ratio_max": 0.20,
        "mandatory_tail_rel_diff_max": 0.40,
        "mandatory_tail_min_samples": 8,
    }

    _, no_guard = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=10.0,
        mandatory_validity={"mandatory_metrics": ["m"]},
        eval_policy={"error_aggregation": {**common_policy, "mandatory_tail_guard_mode": "none"}},
    )
    _, with_guard = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=10.0,
        mandatory_validity={"mandatory_metrics": ["m"]},
        eval_policy={
            "error_aggregation": {
                **common_policy,
                "mandatory_tail_guard_mode": "p95",
                "mandatory_tail_guard_policy": "hard",
            }
        },
    )

    assert no_guard["mandatory_tail_guard_passed"] is True
    assert no_guard["mandatory_error_passed"] is True
    assert with_guard["mandatory_tail_guard_mode_effective"] == "p95"
    assert float(with_guard["mandatory_rel_diff_p95"]) > 0.40
    assert with_guard["mandatory_tail_guard_passed"] is False
    assert with_guard["mandatory_error_passed"] is False
