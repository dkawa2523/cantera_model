from cantera_model.eval.cantera_runner import compare_rows


def test_compare_rows_tiered_allows_optional_low_signal_score_only() -> None:
    baseline = [
        {"case_id": "c1", "m_req": 1.0, "m_opt": 1.0e-20},
        {"case_id": "c2", "m_req": 1.0, "m_opt": 2.0e-20},
    ]
    candidate = [
        {"case_id": "c1", "m_req": 1.0, "m_opt": 1.0e-8},
        {"case_id": "c2", "m_req": 1.0, "m_opt": 1.0e-8},
    ]

    _, summary = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.20,
        mandatory_validity={"mandatory_metrics": ["m_req"]},
        eval_policy={
            "error_aggregation": {
                "mode": "tiered",
                "mandatory_case_pass_min": 1.0,
                "optional_metric_pass_min": 1.0,
                "max_mean_rel_diff_mandatory": 0.2,
                "max_mean_rel_diff_optional": 0.2,
                "optional_weight": 0.5,
            },
            "metric_normalization": {
                "denominator_mode": "max_abs_or_floor",
                "metric_family_abs_floor": {"default": 1.0e-8},
                "low_signal_case_ratio_threshold": 1.0,
                "low_signal_policy": "score_only",
            },
        },
    )

    assert summary["pass_rate"] == 0.0
    assert summary["pass_rate_mandatory_case"] == 1.0
    assert summary["pass_rate_mandatory_case_all_required"] == 1.0
    assert summary["pass_rate_mandatory_case_ratio_mean"] == 1.0
    assert summary["mandatory_case_mode_effective"] == "ratio_mean"
    assert summary["error_gate_passed"] is True
    assert summary["suppressed_low_signal_metric_count"] == 0


def test_compare_rows_tiered_respects_optional_gate_for_non_low_signal() -> None:
    baseline = [
        {"case_id": "c1", "m_req": 1.0, "m_opt": 1.0},
        {"case_id": "c2", "m_req": 1.0, "m_opt": 1.0},
    ]
    candidate = [
        {"case_id": "c1", "m_req": 1.0, "m_opt": 1.5},
        {"case_id": "c2", "m_req": 1.0, "m_opt": 1.5},
    ]

    _, summary = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.20,
        mandatory_validity={"mandatory_metrics": ["m_req"]},
        eval_policy={
            "error_aggregation": {
                "mode": "tiered",
                "mandatory_case_pass_min": 1.0,
                "optional_metric_pass_min": 1.0,
                "max_mean_rel_diff_mandatory": 0.2,
                "max_mean_rel_diff_optional": 0.2,
                "optional_weight": 0.5,
            },
            "metric_normalization": {
                "denominator_mode": "max_abs_or_floor",
                "metric_family_abs_floor": {"default": 1.0e-12},
                "low_signal_case_ratio_threshold": 1.0,
                "low_signal_policy": "score_only",
            },
        },
    )

    assert summary["pass_rate_mandatory_case"] == 1.0
    assert summary["pass_rate_optional_metric_mean"] == 0.0
    assert summary["optional_error_passed"] is False
    assert summary["error_gate_passed"] is False


def test_compare_rows_mandatory_case_mode_ratio_mean_vs_all_required() -> None:
    baseline = [
        {"case_id": "c1", "m1": 1.0, "m2": 1.0, "o": 1.0},
        {"case_id": "c2", "m1": 1.0, "m2": 1.0, "o": 1.0},
    ]
    candidate = [
        {"case_id": "c1", "m1": 1.0, "m2": 1.5, "o": 1.0},  # mandatory ratio 0.5
        {"case_id": "c2", "m1": 1.0, "m2": 1.0, "o": 1.0},  # mandatory ratio 1.0
    ]
    common = {
        "rel_eps": 1.0e-12,
        "rel_tolerance": 0.20,
        "mandatory_validity": {
            "mandatory_metrics": ["m1", "m2"],
            "mandatory_hard_mode": "min_valid_count",
            "min_valid_mandatory_count_abs": 1,
            "min_valid_mandatory_ratio": 0.5,
            "min_valid_mandatory_cap_by_total": True,
        },
    }
    _, s_ratio = compare_rows(
        baseline,
        candidate,
        **common,
        eval_policy={
            "error_aggregation": {
                "mode": "tiered",
                "mandatory_case_mode": "ratio_mean",
                "mandatory_case_pass_min": 0.70,
                "mandatory_quality_scope": "all_units",
            }
        },
    )
    _, s_all = compare_rows(
        baseline,
        candidate,
        **common,
        eval_policy={
            "error_aggregation": {
                "mode": "tiered",
                "mandatory_case_mode": "all_required",
                "mandatory_case_pass_min": 0.70,
                "mandatory_quality_scope": "all_units",
            }
        },
    )

    assert abs(float(s_ratio["pass_rate_mandatory_case_ratio_mean"]) - 0.75) < 1.0e-12
    assert abs(float(s_ratio["pass_rate_mandatory_case_all_required"]) - 0.5) < 1.0e-12
    assert abs(float(s_ratio["pass_rate_mandatory_case"]) - 0.75) < 1.0e-12
    assert s_ratio["mandatory_case_mode_effective"] == "ratio_mean"
    assert s_ratio["mandatory_error_passed"] is True
    assert abs(float(s_all["pass_rate_mandatory_case"]) - 0.5) < 1.0e-12
    assert s_all["mandatory_case_mode_effective"] == "all_required"
    assert s_all["mandatory_error_passed"] is False


def test_compare_rows_mandatory_error_requires_mandatory_validity() -> None:
    baseline = [{"case_id": "c1", "m1": 1.0, "m2": 1.0}]
    candidate = [{"case_id": "c1", "m1": float("nan"), "m2": 1.0}]
    _, summary = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.20,
        mandatory_validity={
            "mandatory_metrics": ["m1", "m2"],
            "mandatory_hard_mode": "min_valid_count",
            "min_valid_mandatory_count_abs": 2,
            "min_valid_mandatory_ratio": 1.0,
            "min_valid_mandatory_cap_by_total": True,
        },
        eval_policy={"error_aggregation": {"mode": "tiered", "mandatory_case_mode": "ratio_mean", "mandatory_case_pass_min": 0.10}},
    )
    assert summary["mandatory_validity_passed"] is False
    assert summary["pass_rate_mandatory_case"] > 0.0
    assert summary["coverage_gate_passed"] is False
    assert summary["mandatory_error_passed"] is True

    _, linked = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.20,
        mandatory_validity={
            "mandatory_metrics": ["m1", "m2"],
            "mandatory_hard_mode": "min_valid_count",
            "min_valid_mandatory_count_abs": 2,
            "min_valid_mandatory_ratio": 1.0,
            "min_valid_mandatory_cap_by_total": True,
        },
        eval_policy={
            "error_aggregation": {
                "mode": "tiered",
                "mandatory_case_mode": "ratio_mean",
                "mandatory_case_pass_min": 0.10,
                "mandatory_error_include_validity": True,
            }
        },
    )
    assert linked["mandatory_error_include_validity_effective"] is True
    assert linked["mandatory_error_passed"] is False


def test_compare_rows_mandatory_mean_family_weighted_reduces_duplicate_family_bias() -> None:
    baseline = [
        {"case_id": "c1", "X_last:A": 1.0, "X_max:A": 1.0, "dep_int:A": 1.0},
        {"case_id": "c2", "X_last:A": 1.0, "X_max:A": 1.0, "dep_int:A": 1.0},
    ]
    candidate = [
        {"case_id": "c1", "X_last:A": 1.8, "X_max:A": 1.1, "dep_int:A": 1.1},
        {"case_id": "c2", "X_last:A": 1.8, "X_max:A": 1.1, "dep_int:A": 1.1},
    ]
    common = {
        "rel_eps": 1.0e-12,
        "rel_tolerance": 1.0,
        "mandatory_validity": {"mandatory_metrics": ["X_last:A", "X_max:A", "dep_int:A"]},
    }
    _, summary_raw = compare_rows(
        baseline,
        candidate,
        **common,
        eval_policy={
            "error_aggregation": {
                "mode": "tiered",
                "mandatory_case_mode": "ratio_mean",
                "mandatory_case_pass_min": 1.0,
                "max_mean_rel_diff_mandatory": 0.25,
                "mandatory_mean_aggregation": "raw",
            }
        },
    )
    _, summary_weighted = compare_rows(
        baseline,
        candidate,
        **common,
        eval_policy={
            "error_aggregation": {
                "mode": "tiered",
                "mandatory_case_mode": "ratio_mean",
                "mandatory_case_pass_min": 1.0,
                "max_mean_rel_diff_mandatory": 0.25,
                "mandatory_mean_aggregation": "family_weighted",
                "mandatory_mean_mode": "raw",
                "mandatory_family_weights": {"X_last": 0.2, "X_max": 0.4, "dep_int": 0.4},
            }
        },
    )

    assert summary_raw["mandatory_mean_aggregation_effective"] == "raw"
    assert summary_weighted["mandatory_mean_aggregation_effective"] == "family_weighted"
    assert float(summary_weighted["mean_rel_diff_mandatory_raw"]) > float(
        summary_weighted["mean_rel_diff_mandatory_family_weighted"]
    )
    assert float(summary_weighted["mean_rel_diff_mandatory"]) == float(
        summary_weighted["mean_rel_diff_mandatory_family_weighted"]
    )
    assert summary_raw["mandatory_error_passed"] is False
    assert summary_weighted["mandatory_error_passed"] is True


def test_compare_rows_mandatory_total_zero_remains_pass() -> None:
    baseline = [{"case_id": "c1", "m": 1.0}]
    candidate = [{"case_id": "c1", "m": 1.0}]
    _, summary = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.20,
        mandatory_validity={"mandatory_metrics": []},
        eval_policy={"error_aggregation": {"mode": "tiered", "mandatory_case_mode": "ratio_mean"}},
    )
    assert summary["mandatory_total_metric_count"] == 0
    assert summary["mandatory_validity_passed"] is True
    assert abs(float(summary["pass_rate_mandatory_case"]) - 1.0) < 1.0e-12
