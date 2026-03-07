from cantera_model.eval.surrogate_eval import compare_with_baseline


def test_compare_with_baseline_surfaces_contract_taxonomy_metadata() -> None:
    baseline = [{"case_id": "c1", "X_last:A": 1.0}]
    candidate = [{"case_id": "c1", "X_last:A": 1.0}]
    _, summary = compare_with_baseline(
        baseline,
        candidate,
        {
            "rel_tolerance": 0.2,
            "rel_eps": 1.0e-12,
            "contract": {"version": "v1"},
            "metric_taxonomy": {"profile": "large_default_v1"},
        },
        qoi_cfg={"species_last": ["A"]},
    )

    assert summary.evaluation_contract_version == "v1"
    assert summary.metric_taxonomy_profile_effective == "large_default_v1"
    assert summary.diagnostic_schema_ok is True
