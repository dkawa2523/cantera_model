from __future__ import annotations

from typing import Any

REQUIRED_TOP_KEYS = (
    "gate_passed",
    "primary_blocker_layer",
    "selected_metrics",
    "gate_evidence",
)

REQUIRED_STAGE_KEYS = (
    "mandatory_validity_passed",
    "coverage_gate_passed",
    "error_gate_passed",
    "mandatory_error_passed",
    "optional_error_passed",
    "error_fail_reason_primary",
    "validity_fail_reason_primary",
    "primary_blocker_layer",
)


def validate_summary_schema(summary: dict[str, Any], *, strict: bool) -> bool:
    missing_top = [key for key in REQUIRED_TOP_KEYS if key not in summary]
    stage = dict((((summary.get("gate_evidence") or {}).get("selected_stage_evidence")) or {}))
    missing_stage = [key for key in REQUIRED_STAGE_KEYS if key not in stage]
    if not missing_top and not missing_stage:
        return True
    msg = "diagnostic schema missing keys"
    if missing_top:
        msg += f" top={missing_top}"
    if missing_stage:
        msg += f" selected_stage_evidence={missing_stage}"
    if strict:
        raise ValueError(msg)
    return False


def project_entry(mode: str, run_id: str, summary: dict[str, Any]) -> dict[str, Any]:
    selected = dict(summary.get("selected_metrics") or {})
    split = dict(summary.get("surrogate_split") or {})
    gate_evidence = dict(summary.get("gate_evidence") or {})
    selected_stage_evidence = dict(gate_evidence.get("selected_stage_evidence") or {})
    fallback_reason = selected.get("learnckpp_fallback_reason") or summary.get("failure_reason")
    return {
        "mode": mode,
        "run_id": run_id,
        "gate_passed": bool(summary.get("gate_passed")),
        "selected_stage": summary.get("selected_stage"),
        "species_before": selected.get("species_before"),
        "species_after": selected.get("species_after"),
        "reactions_before": selected.get("reactions_before"),
        "reactions_after": selected.get("reactions_after"),
        "pass_rate": selected.get("pass_rate"),
        "mean_rel_diff": selected.get("mean_rel_diff"),
        "evaluation_contract_version": summary.get(
            "evaluation_contract_version",
            selected.get(
                "evaluation_contract_version",
                selected_stage_evidence.get("evaluation_contract_version"),
            ),
        ),
        "metric_taxonomy_profile_effective": summary.get(
            "metric_taxonomy_profile_effective",
            selected.get(
                "metric_taxonomy_profile_effective",
                selected_stage_evidence.get("metric_taxonomy_profile_effective"),
            ),
        ),
        "diagnostic_schema_ok": summary.get(
            "diagnostic_schema_ok",
            selected.get("diagnostic_schema_ok", selected_stage_evidence.get("diagnostic_schema_ok")),
        ),
        "metric_drift_raw": selected.get(
            "metric_drift_raw", selected_stage_evidence.get("metric_drift_raw")
        ),
        "metric_drift_effective": selected.get(
            "metric_drift_effective", selected_stage_evidence.get("metric_drift_effective")
        ),
        "metric_drift_effective_cap": selected.get(
            "metric_drift_effective_cap", selected_stage_evidence.get("metric_drift_effective_cap")
        ),
        "selection_quality_score_raw_drift": selected.get(
            "selection_quality_score_raw_drift",
            selected_stage_evidence.get("selection_quality_score_raw_drift"),
        ),
        "compression_refine_applied": selected.get(
            "compression_refine_applied", selected_stage_evidence.get("compression_refine_applied")
        ),
        "compression_refine_trials": selected.get(
            "compression_refine_trials", selected_stage_evidence.get("compression_refine_trials")
        ),
        "compression_refine_reaction_delta": selected.get(
            "compression_refine_reaction_delta",
            selected_stage_evidence.get("compression_refine_reaction_delta"),
        ),
        "compression_refine_species_delta": selected.get(
            "compression_refine_species_delta",
            selected_stage_evidence.get("compression_refine_species_delta"),
        ),
        "compression_refine_mode_effective": selected.get(
            "compression_refine_mode_effective",
            selected_stage_evidence.get("compression_refine_mode_effective"),
        ),
        "compression_refine_guard_passed": selected.get(
            "compression_refine_guard_passed",
            selected_stage_evidence.get("compression_refine_guard_passed"),
        ),
        "pass_rate_mandatory_case": selected.get(
            "pass_rate_mandatory_case", selected_stage_evidence.get("pass_rate_mandatory_case")
        ),
        "pass_rate_mandatory_case_all_units": selected.get(
            "pass_rate_mandatory_case_all_units",
            selected_stage_evidence.get("pass_rate_mandatory_case_all_units"),
        ),
        "pass_rate_mandatory_case_all_required": selected.get(
            "pass_rate_mandatory_case_all_required", selected_stage_evidence.get("pass_rate_mandatory_case_all_required")
        ),
        "pass_rate_mandatory_case_ratio_mean": selected.get(
            "pass_rate_mandatory_case_ratio_mean", selected_stage_evidence.get("pass_rate_mandatory_case_ratio_mean")
        ),
        "pass_rate_mandatory_case_all_required_all_units": selected.get(
            "pass_rate_mandatory_case_all_required_all_units",
            selected_stage_evidence.get("pass_rate_mandatory_case_all_required_all_units"),
        ),
        "pass_rate_mandatory_case_ratio_mean_all_units": selected.get(
            "pass_rate_mandatory_case_ratio_mean_all_units",
            selected_stage_evidence.get("pass_rate_mandatory_case_ratio_mean_all_units"),
        ),
        "pass_rate_all_metric_legacy": selected.get(
            "pass_rate_all_metric_legacy",
            selected_stage_evidence.get("pass_rate_all_metric_legacy", selected.get("pass_rate")),
        ),
        "mandatory_case_mode_effective": selected.get(
            "mandatory_case_mode_effective", selected_stage_evidence.get("mandatory_case_mode_effective")
        ),
        "mandatory_case_unit_weight_mode_effective": selected.get(
            "mandatory_case_unit_weight_mode_effective",
            selected_stage_evidence.get("mandatory_case_unit_weight_mode_effective"),
        ),
        "pass_rate_optional_case": selected.get(
            "pass_rate_optional_case", selected_stage_evidence.get("pass_rate_optional_case")
        ),
        "pass_rate_optional_metric_mean": selected.get(
            "pass_rate_optional_metric_mean", selected_stage_evidence.get("pass_rate_optional_metric_mean")
        ),
        "mean_rel_diff_mandatory": selected.get(
            "mean_rel_diff_mandatory", selected_stage_evidence.get("mean_rel_diff_mandatory")
        ),
        "mean_rel_diff_mandatory_all_units": selected.get(
            "mean_rel_diff_mandatory_all_units",
            selected_stage_evidence.get("mean_rel_diff_mandatory_all_units"),
        ),
        "mean_rel_diff_mandatory_raw": selected.get(
            "mean_rel_diff_mandatory_raw", selected_stage_evidence.get("mean_rel_diff_mandatory_raw")
        ),
        "mean_rel_diff_mandatory_family_weighted": selected.get(
            "mean_rel_diff_mandatory_family_weighted",
            selected_stage_evidence.get("mean_rel_diff_mandatory_family_weighted"),
        ),
        "mean_rel_diff_mandatory_winsorized": selected.get(
            "mean_rel_diff_mandatory_winsorized",
            selected_stage_evidence.get("mean_rel_diff_mandatory_winsorized"),
        ),
        "mandatory_rel_outlier_ratio": selected.get(
            "mandatory_rel_outlier_ratio", selected_stage_evidence.get("mandatory_rel_outlier_ratio")
        ),
        "mandatory_rel_outlier_ratio_all_units": selected.get(
            "mandatory_rel_outlier_ratio_all_units",
            selected_stage_evidence.get("mandatory_rel_outlier_ratio_all_units"),
        ),
        "mandatory_rel_outlier_ratio_max_effective": selected.get(
            "mandatory_rel_outlier_ratio_max_effective",
            selected_stage_evidence.get("mandatory_rel_outlier_ratio_max_effective"),
        ),
        "mandatory_rel_diff_p95": selected.get(
            "mandatory_rel_diff_p95", selected_stage_evidence.get("mandatory_rel_diff_p95")
        ),
        "mandatory_rel_diff_p95_all_units": selected.get(
            "mandatory_rel_diff_p95_all_units",
            selected_stage_evidence.get("mandatory_rel_diff_p95_all_units"),
        ),
        "mandatory_tail_guard_passed": selected.get(
            "mandatory_tail_guard_passed", selected_stage_evidence.get("mandatory_tail_guard_passed")
        ),
        "mandatory_tail_guard_mode_effective": selected.get(
            "mandatory_tail_guard_mode_effective",
            selected_stage_evidence.get("mandatory_tail_guard_mode_effective"),
        ),
        "mandatory_tail_guard_policy_effective": selected.get(
            "mandatory_tail_guard_policy_effective",
            selected_stage_evidence.get("mandatory_tail_guard_policy_effective"),
        ),
        "mandatory_tail_guard_triggered": selected.get(
            "mandatory_tail_guard_triggered",
            selected_stage_evidence.get("mandatory_tail_guard_triggered"),
        ),
        "mandatory_tail_guard_hard_applied": selected.get(
            "mandatory_tail_guard_hard_applied",
            selected_stage_evidence.get("mandatory_tail_guard_hard_applied"),
        ),
        "mandatory_tail_activation_ratio_min_effective": selected.get(
            "mandatory_tail_activation_ratio_min_effective",
            selected_stage_evidence.get("mandatory_tail_activation_ratio_min_effective"),
        ),
        "mandatory_tail_exceed_ref_effective": selected.get(
            "mandatory_tail_exceed_ref_effective",
            selected_stage_evidence.get("mandatory_tail_exceed_ref_effective"),
        ),
        "mandatory_tail_exceed_ratio": selected.get(
            "mandatory_tail_exceed_ratio", selected_stage_evidence.get("mandatory_tail_exceed_ratio")
        ),
        "mandatory_tail_rel_diff_max_effective": selected.get(
            "mandatory_tail_rel_diff_max_effective",
            selected_stage_evidence.get("mandatory_tail_rel_diff_max_effective"),
        ),
        "mandatory_quality_scope_empty": selected.get(
            "mandatory_quality_scope_empty", selected_stage_evidence.get("mandatory_quality_scope_empty")
        ),
        "mandatory_mean_aggregation_effective": selected.get(
            "mandatory_mean_aggregation_effective",
            selected_stage_evidence.get("mandatory_mean_aggregation_effective"),
        ),
        "mandatory_mean_mode_effective": selected.get(
            "mandatory_mean_mode_effective", selected_stage_evidence.get("mandatory_mean_mode_effective")
        ),
        "mean_rel_diff_optional": selected.get(
            "mean_rel_diff_optional", selected_stage_evidence.get("mean_rel_diff_optional")
        ),
        "mean_rel_diff_all_metric_legacy": selected.get(
            "mean_rel_diff_all_metric_legacy",
            selected_stage_evidence.get("mean_rel_diff_all_metric_legacy", selected.get("mean_rel_diff")),
        ),
        "error_gate_score": selected.get("error_gate_score", selected_stage_evidence.get("error_gate_score")),
        "coverage_gate_passed": selected.get(
            "coverage_gate_passed", selected_stage_evidence.get("coverage_gate_passed")
        ),
        "mandatory_quality_passed": selected.get(
            "mandatory_quality_passed", selected_stage_evidence.get("mandatory_quality_passed")
        ),
        "optional_quality_passed": selected.get(
            "optional_quality_passed", selected_stage_evidence.get("optional_quality_passed")
        ),
        "mandatory_error_passed": selected.get(
            "mandatory_error_passed", selected_stage_evidence.get("mandatory_error_passed")
        ),
        "optional_error_passed": selected.get(
            "optional_error_passed", selected_stage_evidence.get("optional_error_passed")
        ),
        "mandatory_error_include_validity_effective": selected.get(
            "mandatory_error_include_validity_effective",
            selected_stage_evidence.get("mandatory_error_include_validity_effective"),
        ),
        "error_fail_reason_primary": summary.get(
            "error_fail_reason_primary",
            selected.get("error_fail_reason_primary", selected_stage_evidence.get("error_fail_reason_primary", "none")),
        ),
        "selection_pool_kind": selected.get(
            "selection_pool_kind", selected_stage_evidence.get("selection_pool_kind", "all")
        ),
        "structure_deficit_score": selected.get(
            "structure_deficit_score", selected_stage_evidence.get("structure_deficit_score", 0.0)
        ),
        "pooling_candidate_count": selected.get(
            "pooling_candidate_count", selected_stage_evidence.get("pooling_candidate_count")
        ),
        "pooling_candidate_unique_count": selected.get(
            "pooling_candidate_unique_count",
            selected_stage_evidence.get("pooling_candidate_unique_count"),
        ),
        "pooling_candidate_selected_backend": selected.get(
            "pooling_candidate_selected_backend",
            selected_stage_evidence.get("pooling_candidate_selected_backend"),
        ),
        "pooling_candidate_selected_source": selected.get(
            "pooling_candidate_selected_source",
            selected_stage_evidence.get("pooling_candidate_selected_source"),
        ),
        "pooling_candidate_selected_coverage_proxy": selected.get(
            "pooling_candidate_selected_coverage_proxy",
            selected_stage_evidence.get("pooling_candidate_selected_coverage_proxy"),
        ),
        "pooling_candidate_selected_dynamics_recon_error": selected.get(
            "pooling_candidate_selected_dynamics_recon_error",
            selected_stage_evidence.get("pooling_candidate_selected_dynamics_recon_error"),
        ),
        "pooling_candidate_selected_max_cluster_size_ratio": selected.get(
            "pooling_candidate_selected_max_cluster_size_ratio",
            selected_stage_evidence.get("pooling_candidate_selected_max_cluster_size_ratio"),
        ),
        "pooling_candidate_scores": selected.get(
            "pooling_candidate_scores", selected_stage_evidence.get("pooling_candidate_scores")
        ),
        "effective_metric_count": selected.get(
            "effective_metric_count", selected_stage_evidence.get("effective_metric_count")
        ),
        "suppressed_low_signal_metric_count": selected.get(
            "suppressed_low_signal_metric_count", selected_stage_evidence.get("suppressed_low_signal_metric_count")
        ),
        "conservation_violation": selected.get("conservation_violation"),
        "hard_ban_violations": summary.get("hard_ban_violations"),
        "split_mode": split.get("mode"),
        "effective_kfolds": split.get("effective_kfolds"),
        "qoi_metrics_count": selected.get("qoi_metrics_count"),
        "integral_qoi_count": selected.get("integral_qoi_count", summary.get("qoi_integral_count")),
        "mandatory_total_metric_count": selected.get(
            "mandatory_total_metric_count", selected_stage_evidence.get("mandatory_total_metric_count")
        ),
        "valid_mandatory_metric_count": selected.get(
            "valid_mandatory_metric_count", selected_stage_evidence.get("valid_mandatory_metric_count")
        ),
        "mandatory_metric_case_pass_rates": selected.get(
            "mandatory_metric_case_pass_rates", selected_stage_evidence.get("mandatory_metric_case_pass_rates")
        ),
        "mandatory_metric_valid_case_pass_min_effective": selected.get(
            "mandatory_metric_valid_case_pass_min_effective",
            selected_stage_evidence.get("mandatory_metric_valid_case_pass_min_effective"),
        ),
        "mandatory_metric_validity_mode_effective": selected.get(
            "mandatory_metric_validity_mode_effective",
            selected_stage_evidence.get("mandatory_metric_validity_mode_effective"),
        ),
        "mandatory_validity_basis_effective": selected.get(
            "mandatory_validity_basis_effective",
            selected_stage_evidence.get("mandatory_validity_basis_effective"),
        ),
        "mandatory_total_gate_unit_count": selected.get(
            "mandatory_total_gate_unit_count", selected_stage_evidence.get("mandatory_total_gate_unit_count")
        ),
        "valid_mandatory_gate_unit_count": selected.get(
            "valid_mandatory_gate_unit_count", selected_stage_evidence.get("valid_mandatory_gate_unit_count")
        ),
        "valid_mandatory_gate_unit_count_case_rate": selected.get(
            "valid_mandatory_gate_unit_count_case_rate",
            selected_stage_evidence.get("valid_mandatory_gate_unit_count_case_rate"),
        ),
        "valid_mandatory_gate_unit_count_coverage": selected.get(
            "valid_mandatory_gate_unit_count_coverage",
            selected_stage_evidence.get("valid_mandatory_gate_unit_count_coverage"),
        ),
        "mandatory_gate_unit_valid_count_shadow_evaluable_ratio": selected.get(
            "mandatory_gate_unit_valid_count_shadow_evaluable_ratio",
            selected_stage_evidence.get("mandatory_gate_unit_valid_count_shadow_evaluable_ratio"),
        ),
        "mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective": selected.get(
            "mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective",
            selected_stage_evidence.get("mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective"),
        ),
        "mandatory_quality_gate_unit_count": selected.get(
            "mandatory_quality_gate_unit_count", selected_stage_evidence.get("mandatory_quality_gate_unit_count")
        ),
        "mandatory_quality_metric_count": selected.get(
            "mandatory_quality_metric_count", selected_stage_evidence.get("mandatory_quality_metric_count")
        ),
        "active_invalid_mandatory_gate_unit_keys": selected.get(
            "active_invalid_mandatory_gate_unit_keys",
            selected_stage_evidence.get("active_invalid_mandatory_gate_unit_keys"),
        ),
        "mandatory_gate_unit_case_pass_rates": selected.get(
            "mandatory_gate_unit_case_pass_rates", selected_stage_evidence.get("mandatory_gate_unit_case_pass_rates")
        ),
        "mandatory_gate_unit_evaluable_case_rates": selected.get(
            "mandatory_gate_unit_evaluable_case_rates",
            selected_stage_evidence.get("mandatory_gate_unit_evaluable_case_rates"),
        ),
        "mandatory_gate_unit_mode_effective": selected.get(
            "mandatory_gate_unit_mode_effective", selected_stage_evidence.get("mandatory_gate_unit_mode_effective")
        ),
        "mandatory_species_family_score_mode_effective": selected.get(
            "mandatory_species_family_score_mode_effective",
            selected_stage_evidence.get("mandatory_species_family_score_mode_effective"),
        ),
        "mandatory_quality_scope_effective": selected.get(
            "mandatory_quality_scope_effective",
            selected_stage_evidence.get("mandatory_quality_scope_effective"),
        ),
        "mandatory_tail_scope_effective": selected.get(
            "mandatory_tail_scope_effective",
            selected_stage_evidence.get("mandatory_tail_scope_effective"),
        ),
        "mandatory_species_family_case_pass_min_effective": selected.get(
            "mandatory_species_family_case_pass_min_effective",
            selected_stage_evidence.get("mandatory_species_family_case_pass_min_effective"),
        ),
        "invalid_mandatory_metric_count": selected.get(
            "invalid_mandatory_metric_count", selected_stage_evidence.get("invalid_mandatory_metric_count")
        ),
        "inactive_mandatory_metric_count": selected.get(
            "inactive_mandatory_metric_count", selected_stage_evidence.get("inactive_mandatory_metric_count")
        ),
        "active_invalid_mandatory_metric_count": selected.get(
            "active_invalid_mandatory_metric_count", selected_stage_evidence.get("active_invalid_mandatory_metric_count")
        ),
        "min_valid_mandatory_count_effective": selected.get(
            "min_valid_mandatory_count_effective", selected_stage_evidence.get("min_valid_mandatory_count_effective")
        ),
        "mandatory_validity_passed": selected.get(
            "mandatory_validity_passed", selected_stage_evidence.get("mandatory_validity_passed")
        ),
        "metric_clip_guardrail_trigger_ratio": selected.get(
            "metric_clip_guardrail_trigger_ratio", selected_stage_evidence.get("metric_clip_guardrail_trigger_ratio")
        ),
        "replay_health_trust_invalid": selected.get(
            "replay_health_trust_invalid", selected_stage_evidence.get("replay_health_trust_invalid")
        ),
        "structure_feedback_multiplier": selected.get(
            "structure_feedback_multiplier", selected_stage_evidence.get("structure_feedback_multiplier")
        ),
        "primary_blocker_layer": summary.get(
            "primary_blocker_layer",
            selected.get("primary_blocker_layer", selected_stage_evidence.get("primary_blocker_layer", "none")),
        ),
        "secondary_blockers": summary.get(
            "secondary_blockers",
            selected.get("secondary_blockers", selected_stage_evidence.get("secondary_blockers", [])),
        ),
        "validity_fail_reason_primary": summary.get(
            "validity_fail_reason_primary",
            selected.get("validity_fail_reason_primary", selected_stage_evidence.get("validity_fail_reason_primary", "none")),
        ),
        "timing_total_s": selected.get("timing_total_s", summary.get("timing_total_s")),
        "timing_stage_s": selected.get("timing_stage_s", selected_stage_evidence.get("timing_stage_s")),
        "timing_pooling_fit_s": selected.get("timing_pooling_fit_s", selected_stage_evidence.get("timing_pooling_fit_s")),
        "timing_bridge_s": selected.get("timing_bridge_s", selected_stage_evidence.get("timing_bridge_s")),
        "timing_surrogate_eval_s": selected.get("timing_surrogate_eval_s", selected_stage_evidence.get("timing_surrogate_eval_s")),
        "timing_physical_gate_s": selected.get("timing_physical_gate_s", selected_stage_evidence.get("timing_physical_gate_s")),
        "timing_projection_s": selected.get("timing_projection_s", selected_stage_evidence.get("timing_projection_s")),
        "config_hash": summary.get("config_hash"),
        "git_commit": summary.get("git_commit"),
        "pid": summary.get("pid"),
        "started_at": summary.get("started_at"),
        "finished_at": summary.get("finished_at"),
        "fallback_reason": fallback_reason,
    }


SUMMARY_EXPORT_KEYS = tuple(project_entry("", "", {}).keys())
