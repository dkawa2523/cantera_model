from cantera_model.eval.cantera_runner import compare_rows


def test_compare_rows_metric_taxonomy_profile_override_changes_floor_handling() -> None:
    baseline = [{"case_id": "c1", "M_LAST:A": 1.0e-10}]
    candidate = [{"case_id": "c1", "M_LAST:A": 2.0e-10}]
    mandatory = {"mandatory_metrics": ["M_LAST:A"]}

    _, legacy = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.20,
        mandatory_validity=mandatory,
        eval_policy={"error_aggregation": {"mode": "tiered", "mandatory_quality_scope": "all_units"}},
    )
    _, profiled = compare_rows(
        baseline,
        candidate,
        rel_eps=1.0e-12,
        rel_tolerance=0.20,
        mandatory_validity=mandatory,
        eval_policy={
            "error_aggregation": {"mode": "tiered", "mandatory_quality_scope": "all_units"},
            "metric_taxonomy": {
                "profile": "large_default_v1",
                "family_prefix": {"M_LAST:": "X_last"},
                "species_token": {"delimiter": ":", "take": "after_first"},
                "metric_family_abs_floor": {"X_last": 1.0e-8, "default": 1.0e-12},
            },
        },
    )

    assert float(legacy["pass_rate_mandatory_case"]) == 0.0
    assert float(profiled["pass_rate_mandatory_case"]) == 1.0
    assert profiled["metric_taxonomy_profile_effective"] == "large_default_v1"
