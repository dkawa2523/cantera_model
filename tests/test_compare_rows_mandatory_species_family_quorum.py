from cantera_model.eval.cantera_runner import compare_rows


def test_mandatory_species_family_quorum_reduces_duplicate_family_penalty() -> None:
    baseline = [
        {"case_id": "c1", "X_last:A": 1.0, "X_max:A": 1.0, "X_int:A": 1.0, "X_last:B": 1.0},
        {"case_id": "c2", "X_last:A": 1.0, "X_max:A": 1.0, "X_int:A": 1.0, "X_last:B": 1.0},
    ]
    candidate = [
        {"case_id": "c1", "X_last:A": 1.30, "X_max:A": 1.0, "X_int:A": 1.0, "X_last:B": 1.0},
        {"case_id": "c2", "X_last:A": 1.0, "X_max:A": 1.30, "X_int:A": 1.0, "X_last:B": 1.0},
    ]
    common_mandatory = {
        "mandatory_metrics": ["X_last:A", "X_max:A", "X_int:A", "X_last:B"],
        "mandatory_metric_validity_mode": "case_pass_rate",
        "mandatory_metric_case_pass_min": 0.75,
        "mandatory_hard_mode": "min_valid_count",
        "min_valid_mandatory_count_abs": 3,
        "min_valid_mandatory_ratio": 1.0,
        "min_valid_mandatory_cap_by_total": True,
    }

    _, metric_summary = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.20,
        mandatory_validity={
            **common_mandatory,
            "mandatory_valid_unit_mode": "metric",
            "mandatory_species_family_case_pass_min": 0.66,
        },
        eval_policy={"error_aggregation": {"mode": "tiered", "mandatory_case_pass_min": 0.0}},
    )
    _, quorum_summary = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.20,
        mandatory_validity={
            **common_mandatory,
            "mandatory_valid_unit_mode": "species_family_quorum",
            "mandatory_species_family_case_pass_min": 0.66,
        },
        eval_policy={"error_aggregation": {"mode": "tiered", "mandatory_case_pass_min": 0.0}},
    )

    assert metric_summary["mandatory_gate_unit_mode_effective"] == "metric"
    assert metric_summary["mandatory_total_gate_unit_count"] == 4
    assert metric_summary["valid_mandatory_gate_unit_count"] == 4
    assert metric_summary["mandatory_validity_passed"] is True

    assert quorum_summary["mandatory_gate_unit_mode_effective"] == "species_family_quorum"
    assert quorum_summary["mandatory_total_gate_unit_count"] == 2
    assert quorum_summary["valid_mandatory_gate_unit_count"] == 2
    assert quorum_summary["mandatory_validity_passed"] is True
    assert float(quorum_summary["pass_rate_mandatory_case_ratio_mean"]) >= float(
        metric_summary["pass_rate_mandatory_case_ratio_mean"]
    )


def test_mandatory_species_family_quorum_supports_weighted_family_score() -> None:
    baseline = [
        {"case_id": "c1", "X_last:A": 1.0, "X_max:A": 1.0, "X_int:A": 1.0},
        {"case_id": "c2", "X_last:A": 1.0, "X_max:A": 1.0, "X_int:A": 1.0},
    ]
    candidate = [
        {"case_id": "c1", "X_last:A": 1.0, "X_max:A": 1.0, "X_int:A": 1.30},
        {"case_id": "c2", "X_last:A": 1.0, "X_max:A": 1.0, "X_int:A": 1.0},
    ]
    mandatory_validity = {
        "mandatory_metrics": ["X_last:A", "X_max:A", "X_int:A"],
        "mandatory_valid_unit_mode": "species_family_quorum",
        "mandatory_species_family_case_pass_min": 0.75,
        "mandatory_hard_mode": "min_valid_count",
        "min_valid_mandatory_count_abs": 0,
        "min_valid_mandatory_ratio": 0.0,
        "min_valid_mandatory_cap_by_total": True,
    }
    eval_policy = {
        "error_aggregation": {
            "mode": "tiered",
            "mandatory_case_pass_min": 0.75,
            "mandatory_quality_scope": "all_units",
            "optional_metric_pass_min": 0.0,
            "max_mean_rel_diff_mandatory": 10.0,
            "max_mean_rel_diff_optional": 10.0,
            "mandatory_family_weights": {"X_last": 0.6, "X_max": 0.3, "X_int": 0.1, "default": 1.0},
        }
    }

    _, uniform_summary = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.20,
        mandatory_validity={
            **mandatory_validity,
            "mandatory_species_family_score_mode": "uniform",
        },
        eval_policy=eval_policy,
    )
    _, weighted_summary = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.20,
        mandatory_validity={
            **mandatory_validity,
            "mandatory_species_family_score_mode": "weighted",
        },
        eval_policy=eval_policy,
    )

    assert weighted_summary["mandatory_species_family_score_mode_effective"] == "weighted"
    assert uniform_summary["mandatory_species_family_score_mode_effective"] == "uniform"
    assert float(weighted_summary["pass_rate_mandatory_case_ratio_mean"]) > float(
        uniform_summary["pass_rate_mandatory_case_ratio_mean"]
    )
    assert uniform_summary["error_fail_reason_primary"] == "mandatory_case_rate"
    assert weighted_summary["error_fail_reason_primary"] == "none"


def test_mandatory_case_rate_supports_family_weighted_gate_units() -> None:
    baseline = [
        {"case_id": "c1", "X_last:A": 1.0, "X_max:B": 1.0, "X_max:C": 1.0},
        {"case_id": "c2", "X_last:A": 1.0, "X_max:B": 1.0, "X_max:C": 1.0},
    ]
    candidate = [
        {"case_id": "c1", "X_last:A": 1.0, "X_max:B": 1.30, "X_max:C": 1.30},
        {"case_id": "c2", "X_last:A": 1.0, "X_max:B": 1.0, "X_max:C": 1.0},
    ]
    mandatory_validity = {
        "mandatory_metrics": ["X_last:A", "X_max:B", "X_max:C"],
        "mandatory_valid_unit_mode": "species_family_quorum",
        "mandatory_species_family_score_mode": "weighted",
        "mandatory_species_family_case_pass_min": 0.67,
        "mandatory_hard_mode": "min_valid_count",
        "min_valid_mandatory_count_abs": 0,
        "min_valid_mandatory_ratio": 0.0,
        "min_valid_mandatory_cap_by_total": True,
    }
    common_error_policy = {
        "mode": "tiered",
        "mandatory_case_pass_min": 0.75,
        "mandatory_quality_scope": "all_units",
        "optional_metric_pass_min": 0.0,
        "max_mean_rel_diff_mandatory": 10.0,
        "max_mean_rel_diff_optional": 10.0,
        "mandatory_family_weights": {"X_last": 0.8, "X_max": 0.1, "default": 1.0},
    }

    _, uniform_case_summary = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.20,
        mandatory_validity=mandatory_validity,
        eval_policy={
            "error_aggregation": {
                **common_error_policy,
                "mandatory_case_unit_weight_mode": "uniform",
            }
        },
    )
    _, weighted_case_summary = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.20,
        mandatory_validity=mandatory_validity,
        eval_policy={
            "error_aggregation": {
                **common_error_policy,
                "mandatory_case_unit_weight_mode": "family_weighted",
            }
        },
    )

    assert uniform_case_summary["mandatory_case_unit_weight_mode_effective"] == "uniform"
    assert weighted_case_summary["mandatory_case_unit_weight_mode_effective"] == "family_weighted"
    assert float(weighted_case_summary["pass_rate_mandatory_case_ratio_mean"]) > float(
        uniform_case_summary["pass_rate_mandatory_case_ratio_mean"]
    )
    assert uniform_case_summary["mandatory_quality_passed"] is False
    assert weighted_case_summary["mandatory_quality_passed"] is True
