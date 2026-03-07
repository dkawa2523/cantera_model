import pytest

from cantera_model.eval.cantera_runner import compare_rows


def test_compare_rows_mandatory_validity_case_pass_rate_mode() -> None:
    baseline = [
        {"case_id": "c1", "m1": 1.0, "m2": 1.0, "m3": 1.0},
        {"case_id": "c2", "m1": 1.0, "m2": 1.0, "m3": 1.0},
    ]
    candidate = [
        {"case_id": "c1", "m1": 1.0, "m2": 1.4, "m3": 1.0},  # m2 fail
        {"case_id": "c2", "m1": 1.0, "m2": 1.0, "m3": 1.0},
    ]

    _, summary = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.20,
        mandatory_validity={
            "mandatory_metrics": ["m1", "m2", "m3"],
            "mandatory_metric_validity_mode": "case_pass_rate",
            "mandatory_metric_case_pass_min": 0.50,
            "mandatory_hard_mode": "min_valid_count",
            "min_valid_mandatory_count_abs": 3,
            "min_valid_mandatory_ratio": 1.0,
            "min_valid_mandatory_cap_by_total": True,
        },
        eval_policy={"error_aggregation": {"mode": "tiered"}},
    )

    assert summary["mandatory_metric_validity_mode_effective"] == "case_pass_rate"
    assert abs(float(summary["mandatory_metric_case_pass_rates"]["m2"]) - 0.5) < 1.0e-12
    assert summary["valid_mandatory_metric_count"] == 3
    assert summary["mandatory_validity_passed"] is True


def test_compare_rows_mandatory_validity_all_cases_mode() -> None:
    baseline = [
        {"case_id": "c1", "m1": 1.0, "m2": 1.0, "m3": 1.0},
        {"case_id": "c2", "m1": 1.0, "m2": 1.0, "m3": 1.0},
    ]
    candidate = [
        {"case_id": "c1", "m1": 1.0, "m2": 1.4, "m3": 1.0},  # m2 fail
        {"case_id": "c2", "m1": 1.0, "m2": 1.0, "m3": 1.0},
    ]

    _, summary = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.20,
        mandatory_validity={
            "mandatory_metrics": ["m1", "m2", "m3"],
            "mandatory_metric_validity_mode": "all_cases",
            "mandatory_hard_mode": "min_valid_count",
            "min_valid_mandatory_count_abs": 3,
            "min_valid_mandatory_ratio": 1.0,
            "min_valid_mandatory_cap_by_total": True,
        },
        eval_policy={"error_aggregation": {"mode": "tiered"}},
    )

    assert summary["mandatory_metric_validity_mode_effective"] == "all_cases"
    assert summary["active_invalid_mandatory_metric_count"] == 1
    assert summary["valid_mandatory_metric_count"] == 2
    assert summary["mandatory_validity_passed"] is True


def test_compare_rows_mandatory_validity_basis_case_pass_rate_removed() -> None:
    baseline = [{"case_id": "c1", "m1": 1.0}]
    candidate = [{"case_id": "c1", "m1": 1.0}]
    with pytest.raises(ValueError, match="mandatory_validity_basis"):
        compare_rows(
            baseline,
            candidate,
            rel_eps=1.0e-12,
            rel_tolerance=0.20,
            mandatory_validity={
                "mandatory_metrics": ["m1"],
                "mandatory_validity_basis": "case_pass_rate",
            },
            eval_policy={"error_aggregation": {"mode": "tiered"}},
        )
