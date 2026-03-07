from cantera_model.eval.surrogate_eval import compare_with_baseline, run_surrogate_cases


def test_run_surrogate_cases_can_disable_builtin_qoi_metrics() -> None:
    rows = run_surrogate_cases(
        {"global_scale": 1.0},
        [{"case_id": "c1", "T0": 1000.0, "P0_atm": 1.0, "phi": 1.0, "t_end": 0.1}],
        {
            "species_last": ["A"],
            "qoi_builtin_metrics": {
                "include_temperature_metrics": False,
                "include_ignition_delay": False,
            },
        },
    )
    assert len(rows) == 1
    row = rows[0]
    assert "X_last:A" in row
    assert "ignition_delay" not in row
    assert "T_max" not in row
    assert "T_last" not in row


def test_compare_with_baseline_defaults_mandatory_to_explicit_qoi() -> None:
    baseline = [
        {
            "case_id": "c1",
            "X_last:A": 1.0,
            "ignition_delay": 0.1,
            "T_max": 1000.0,
            "T_last": 900.0,
        }
    ]
    candidate = [dict(baseline[0])]
    _, summary = compare_with_baseline(
        baseline,
        candidate,
        {"rel_tolerance": 0.2, "rel_eps": 1.0e-12},
        qoi_cfg={"species_last": ["A"]},
    )

    assert summary.mandatory_total_metric_count == 1
    assert summary.mandatory_total_gate_unit_count == 1
    assert summary.valid_mandatory_gate_unit_count == 1
    assert summary.valid_mandatory_gate_unit_count_case_rate == 1
    assert summary.valid_mandatory_gate_unit_count_coverage == 1
    assert summary.mandatory_quality_gate_unit_count == 1
    assert summary.mandatory_quality_metric_count == 1
    assert summary.mandatory_gate_unit_mode_effective == "species_family_quorum"
    assert summary.mandatory_validity_basis_effective == "coverage_evaluable"
    assert summary.mandatory_gate_unit_evaluable_case_rates == {"A": 1.0}
    assert summary.mandatory_quality_scope_effective == "valid_only"
    assert summary.mandatory_tail_scope_effective == "quality_scope"
    assert summary.pass_rate_mandatory_case == 1.0
    assert summary.pass_rate_mandatory_case_all_units == 1.0
    assert summary.coverage_gate_passed is True
    assert summary.mandatory_quality_passed is True
    assert summary.optional_quality_passed is True
    assert summary.error_fail_reason_primary == "none"
    assert summary.mandatory_tail_guard_mode_effective == "p95"
    assert summary.mandatory_tail_guard_passed is True
    assert summary.mandatory_quality_scope_empty is False
    assert summary.error_gate_passed is True
    assert summary.mandatory_mean_aggregation_effective == "raw"
