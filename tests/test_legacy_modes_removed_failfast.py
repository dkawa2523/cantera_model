import pytest

from cantera_model.eval.cantera_runner import compare_rows


def test_compare_rows_rejects_legacy_error_aggregation_mode() -> None:
    baseline = [{"case_id": "c1", "m": 1.0}]
    candidate = [{"case_id": "c1", "m": 1.0}]
    with pytest.raises(ValueError, match="error_aggregation.mode"):
        compare_rows(
            baseline,
            candidate,
            rel_eps=1.0e-12,
            rel_tolerance=0.2,
            mandatory_validity={"mandatory_metrics": ["m"]},
            eval_policy={"error_aggregation": {"mode": "legacy_all_metric"}},
        )


def test_compare_rows_rejects_legacy_denominator_mode() -> None:
    baseline = [{"case_id": "c1", "m": 1.0}]
    candidate = [{"case_id": "c1", "m": 1.0}]
    with pytest.raises(ValueError, match="denominator_mode"):
        compare_rows(
            baseline,
            candidate,
            rel_eps=1.0e-12,
            rel_tolerance=0.2,
            mandatory_validity={"mandatory_metrics": ["m"]},
            eval_policy={
                "error_aggregation": {
                    "mode": "tiered",
                    "mandatory_case_pass_min": 0.75,
                    "optional_metric_pass_min": 0.75,
                    "max_mean_rel_diff_mandatory": 0.4,
                    "max_mean_rel_diff_optional": 0.6,
                },
                "metric_normalization": {"denominator_mode": "legacy"},
            },
        )


def test_compare_rows_rejects_case_pass_rate_validity_basis() -> None:
    baseline = [{"case_id": "c1", "m": 1.0}]
    candidate = [{"case_id": "c1", "m": 1.0}]
    with pytest.raises(ValueError, match="mandatory_validity_basis"):
        compare_rows(
            baseline,
            candidate,
            rel_eps=1.0e-12,
            rel_tolerance=0.2,
            mandatory_validity={
                "mandatory_metrics": ["m"],
                "mandatory_validity_basis": "case_pass_rate",
            },
            eval_policy={
                "error_aggregation": {
                    "mode": "tiered",
                    "mandatory_case_pass_min": 0.75,
                    "optional_metric_pass_min": 0.75,
                    "max_mean_rel_diff_mandatory": 0.4,
                    "max_mean_rel_diff_optional": 0.6,
                }
            },
        )
