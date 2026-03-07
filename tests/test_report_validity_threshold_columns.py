import csv

from cantera_model.reporting.report import write_report


def test_report_emits_validity_threshold_columns(tmp_path) -> None:
    rows = [
        {
            "stage": "A",
            "species_after": 12,
            "reactions_after": 8,
            "overall_candidates": 40,
            "overall_selected": 8,
            "overall_select_ratio": 0.2,
            "mean_rel_diff": 0.12,
            "gate_passed": True,
            "floor_passed": True,
            "balance_gate_passed": True,
            "cluster_guard_passed": True,
            "weighted_active_species_coverage": 0.84,
            "active_species_coverage_top_weighted": 0.86,
            "essential_species_coverage": 0.88,
            "max_cluster_size_ratio": 0.32,
            "balance_mode": "hybrid",
            "selection_score": 1.2,
            "rs_upper_effective": 5.5,
            "active_cov_effective_floor": 0.40,
            "balance_margin": 0.03,
            "balance_dynamic_applied": True,
            "mandatory_total_metric_count": 3,
            "valid_mandatory_metric_count": 2,
            "inactive_mandatory_metric_count": 1,
            "active_invalid_mandatory_metric_count": 0,
            "mandatory_total_gate_unit_count": 2,
            "valid_mandatory_gate_unit_count": 2,
            "valid_mandatory_gate_unit_count_case_rate": 1,
            "valid_mandatory_gate_unit_count_coverage": 2,
            "mandatory_validity_basis_effective": "coverage_evaluable",
            "mandatory_quality_gate_unit_count": 2,
            "mandatory_quality_metric_count": 2,
            "active_invalid_mandatory_gate_unit_keys": ["A"],
            "mandatory_gate_unit_case_pass_rates": {"A": 1.0, "B": 1.0},
            "mandatory_gate_unit_evaluable_case_rates": {"A": 1.0, "B": 1.0},
            "mandatory_gate_unit_mode_effective": "species_family_quorum",
            "mandatory_species_family_score_mode_effective": "weighted",
            "mandatory_quality_scope_effective": "valid_only",
            "mandatory_tail_scope_effective": "quality_scope",
            "mandatory_species_family_case_pass_min_effective": 0.67,
            "mandatory_validity_passed": True,
            "pass_rate_mandatory_case": 0.8,
            "pass_rate_mandatory_case_all_units": 0.6,
            "pass_rate_mandatory_case_all_required": 0.5,
            "pass_rate_mandatory_case_ratio_mean": 0.8,
            "pass_rate_mandatory_case_all_required_all_units": 0.4,
            "pass_rate_mandatory_case_ratio_mean_all_units": 0.6,
            "pass_rate_all_metric_legacy": 0.2,
            "mandatory_case_mode_effective": "ratio_mean",
            "mandatory_case_unit_weight_mode_effective": "family_weighted",
            "pass_rate_optional_case": 0.7,
            "pass_rate_optional_metric_mean": 0.75,
            "mean_rel_diff_mandatory": 0.2,
            "mean_rel_diff_mandatory_all_units": 0.45,
            "mean_rel_diff_mandatory_raw": 0.24,
            "mean_rel_diff_mandatory_family_weighted": 0.2,
            "mean_rel_diff_mandatory_winsorized": 0.21,
            "mandatory_rel_outlier_ratio": 0.05,
            "mandatory_rel_outlier_ratio_all_units": 0.10,
            "mandatory_rel_diff_p95": 1.2,
            "mandatory_rel_diff_p95_all_units": 1.8,
            "mandatory_tail_guard_passed": True,
            "mandatory_tail_guard_triggered": True,
            "mandatory_tail_guard_hard_applied": False,
            "mandatory_tail_guard_mode_effective": "p95",
            "mandatory_tail_guard_policy_effective": "conditional_hard",
            "mandatory_tail_activation_ratio_min_effective": 0.10,
            "mandatory_tail_exceed_ref_effective": "tail_max",
            "mandatory_tail_exceed_ratio": 0.05,
            "mandatory_tail_rel_diff_max_effective": 1.5,
            "mandatory_quality_scope_empty": False,
            "mandatory_mean_aggregation_effective": "family_weighted",
            "mandatory_mean_mode_effective": "winsorized",
            "mean_rel_diff_optional": 0.3,
            "mean_rel_diff_all_metric_legacy": 0.8,
            "error_gate_score": 0.78,
            "coverage_gate_passed": True,
            "mandatory_quality_passed": True,
            "optional_quality_passed": True,
            "mandatory_error_passed": True,
            "optional_error_passed": True,
            "mandatory_error_include_validity_effective": False,
            "error_fail_reason_primary": "none",
            "selection_pool_kind": "floor",
            "structure_deficit_score": 0.2,
            "pooling_candidate_count": 2,
            "pooling_candidate_unique_count": 2,
            "pooling_candidate_selected_backend": "pyg",
            "pooling_candidate_selected_source": "swap_refine",
            "pooling_candidate_selected_coverage_proxy": 0.66,
            "pooling_candidate_selected_dynamics_recon_error": 0.19,
            "pooling_candidate_selected_max_cluster_size_ratio": 0.41,
            "pooling_candidate_scores": [{"backend": "pyg"}, {"backend": "numpy"}],
            "effective_metric_count": 10,
            "suppressed_low_signal_metric_count": 1,
            "metric_clip_guardrail_trigger_ratio": 0.02,
            "replay_health_trust_invalid": False,
            "structure_feedback_multiplier": 1.10,
            "metric_drift_raw": 1.72,
            "metric_drift_effective_cap": 1.30,
            "selection_quality_score_raw_drift": 0.64,
            "compression_refine_applied": True,
            "compression_refine_trials": 2,
            "compression_refine_reaction_delta": 3,
            "compression_refine_species_delta": 0,
            "compression_refine_mode_effective": "baseline_grid",
            "compression_refine_guard_passed": True,
            "mandatory_gate_unit_valid_count_shadow_evaluable_ratio": 1,
            "mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective": 0.25,
            "primary_blocker_layer": "error",
            "validity_fail_reason_primary": "none",
            "timing_stage_s": 0.12,
            "timing_bridge_s": 0.07,
            "timing_surrogate_eval_s": 0.03,
            "timing_physical_gate_s": 0.02,
            "timing_projection_s": 0.01,
        }
    ]
    out = write_report(
        tmp_path / "validity_report",
        run_id="validity_report",
        stage_rows=rows,
        selected_stage="A",
        summary_payload={
            "gate_passed": True,
            "hard_ban_violations": 0,
            "diagnostic_schema_ok": True,
            "evaluation_contract_version": "v1",
            "metric_taxonomy_profile_effective": "large_default_v1",
            "primary_blocker_layer": "error",
            "validity_fail_reason_primary": "none",
            "timing_total_s": 0.34,
            "selected_metrics": {
                "compression_refine_applied": True,
                "compression_refine_trials": 2,
                "compression_refine_reaction_delta": 3,
                "compression_refine_mode_effective": "baseline_grid",
                "compression_refine_guard_passed": True,
            },
            "reduction_trace": {"candidate_trend": rows},
        },
    )
    with (out / "metrics.csv").open("r", newline="") as handle:
        metrics = list(csv.DictReader(handle))
    report_md = (out / "report.md").read_text()

    assert "mandatory_total_metric_count" in metrics[0]
    assert "valid_mandatory_metric_count" in metrics[0]
    assert "mandatory_total_gate_unit_count" in metrics[0]
    assert "valid_mandatory_gate_unit_count" in metrics[0]
    assert "valid_mandatory_gate_unit_count_case_rate" in metrics[0]
    assert "valid_mandatory_gate_unit_count_coverage" in metrics[0]
    assert "mandatory_validity_basis_effective" in metrics[0]
    assert "mandatory_quality_gate_unit_count" in metrics[0]
    assert "mandatory_quality_metric_count" in metrics[0]
    assert "active_invalid_mandatory_gate_unit_keys" in metrics[0]
    assert "mandatory_gate_unit_mode_effective" in metrics[0]
    assert "mandatory_species_family_score_mode_effective" in metrics[0]
    assert "mandatory_quality_scope_effective" in metrics[0]
    assert "mandatory_tail_scope_effective" in metrics[0]
    assert "mandatory_species_family_case_pass_min_effective" in metrics[0]
    assert "inactive_mandatory_metric_count" in metrics[0]
    assert "active_invalid_mandatory_metric_count" in metrics[0]
    assert "pass_rate_mandatory_case" in metrics[0]
    assert "pass_rate_mandatory_case_all_units" in metrics[0]
    assert "pass_rate_mandatory_case_all_required" in metrics[0]
    assert "pass_rate_mandatory_case_ratio_mean" in metrics[0]
    assert "pass_rate_mandatory_case_all_required_all_units" in metrics[0]
    assert "pass_rate_mandatory_case_ratio_mean_all_units" in metrics[0]
    assert "pass_rate_all_metric_legacy" in metrics[0]
    assert "mandatory_case_mode_effective" in metrics[0]
    assert "mandatory_case_unit_weight_mode_effective" in metrics[0]
    assert "pass_rate_optional_case" in metrics[0]
    assert "pass_rate_optional_metric_mean" in metrics[0]
    assert "mean_rel_diff_mandatory_raw" in metrics[0]
    assert "mean_rel_diff_mandatory_all_units" in metrics[0]
    assert "mean_rel_diff_mandatory_family_weighted" in metrics[0]
    assert "mean_rel_diff_mandatory_winsorized" in metrics[0]
    assert "mandatory_rel_outlier_ratio" in metrics[0]
    assert "mandatory_rel_outlier_ratio_all_units" in metrics[0]
    assert "mandatory_rel_diff_p95" in metrics[0]
    assert "mandatory_rel_diff_p95_all_units" in metrics[0]
    assert "mandatory_tail_guard_passed" in metrics[0]
    assert "mandatory_tail_guard_triggered" in metrics[0]
    assert "mandatory_tail_guard_hard_applied" in metrics[0]
    assert "mandatory_tail_guard_mode_effective" in metrics[0]
    assert "mandatory_tail_guard_policy_effective" in metrics[0]
    assert "mandatory_tail_activation_ratio_min_effective" in metrics[0]
    assert "mandatory_tail_exceed_ref_effective" in metrics[0]
    assert "mandatory_tail_exceed_ratio" in metrics[0]
    assert "mandatory_tail_rel_diff_max_effective" in metrics[0]
    assert "mandatory_quality_scope_empty" in metrics[0]
    assert "mandatory_mean_aggregation_effective" in metrics[0]
    assert "mandatory_mean_mode_effective" in metrics[0]
    assert "mean_rel_diff_all_metric_legacy" in metrics[0]
    assert "coverage_gate_passed" in metrics[0]
    assert "mandatory_quality_passed" in metrics[0]
    assert "optional_quality_passed" in metrics[0]
    assert "mandatory_error_passed" in metrics[0]
    assert "optional_error_passed" in metrics[0]
    assert "mandatory_error_include_validity_effective" in metrics[0]
    assert "error_fail_reason_primary" in metrics[0]
    assert "error_gate_score" in metrics[0]
    assert "effective_metric_count" in metrics[0]
    assert "suppressed_low_signal_metric_count" in metrics[0]
    assert "metric_clip_guardrail_trigger_ratio" in metrics[0]
    assert "structure_feedback_multiplier" in metrics[0]
    assert "metric_drift_raw" in metrics[0]
    assert "metric_drift_effective_cap" in metrics[0]
    assert "selection_quality_score_raw_drift" in metrics[0]
    assert "compression_refine_applied" in metrics[0]
    assert "compression_refine_trials" in metrics[0]
    assert "compression_refine_reaction_delta" in metrics[0]
    assert "compression_refine_species_delta" in metrics[0]
    assert "compression_refine_mode_effective" in metrics[0]
    assert "compression_refine_guard_passed" in metrics[0]
    assert "mandatory_gate_unit_valid_count_shadow_evaluable_ratio" in metrics[0]
    assert "mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective" in metrics[0]
    assert "primary_blocker_layer" in metrics[0]
    assert "validity_fail_reason_primary" in metrics[0]
    assert "selection_pool_kind" in metrics[0]
    assert "structure_deficit_score" in metrics[0]
    assert "pooling_candidate_count" in metrics[0]
    assert "pooling_candidate_unique_count" in metrics[0]
    assert "pooling_candidate_selected_backend" in metrics[0]
    assert "pooling_candidate_selected_source" in metrics[0]
    assert "pooling_candidate_selected_coverage_proxy" in metrics[0]
    assert "pooling_candidate_selected_dynamics_recon_error" in metrics[0]
    assert "pooling_candidate_selected_max_cluster_size_ratio" in metrics[0]
    assert "pooling_candidate_scores" in metrics[0]
    assert "timing_stage_s" in metrics[0]
    assert "timing_bridge_s" in metrics[0]
    assert "timing_projection_s" in metrics[0]
    assert "| valid_mandatory | mandatory_total | inactive_mandatory | active_invalid_mandatory |" in report_md
    assert "| mand_case_pass | mand_case_all | mand_case_ratio | mand_case_mode | opt_case_pass | opt_metric_pass |" in report_md
    assert "| drift_raw | drift_cap | sel_quality_raw_drift | valid_shadow_evaluable | shadow_ratio_min |" in report_md
    assert "| t_stage_s | t_bridge_s | t_surr_s | t_phys_s | t_proj_s |" in report_md
    assert "- diagnostic_schema_ok: True" in report_md
    assert "- evaluation_contract_version: v1" in report_md
    assert "- metric_taxonomy_profile_effective: large_default_v1" in report_md
    assert "- compression_refine_applied: True" in report_md
    assert "- compression_refine_trials: 2" in report_md
    assert "- compression_refine_reaction_delta: 3" in report_md
    assert "- compression_refine_mode_effective: baseline_grid" in report_md
    assert "- pooling_candidate_count: 2" in report_md
    assert "- pooling_candidate_unique_count: 2" in report_md
    assert "- pooling_candidate_selected_backend: pyg" in report_md
    assert "- pooling_candidate_selected_source: swap_refine" in report_md
    assert "- pooling_candidate_selected_dynamics_recon_error: 0.190000" in report_md
