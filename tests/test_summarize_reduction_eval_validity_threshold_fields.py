import json
import subprocess
import sys
from pathlib import Path


def test_summarize_reduction_eval_emits_validity_threshold_fields(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    out_dir = reports / "r0"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "gate_passed": False,
        "selected_stage": "C",
        "hard_ban_violations": 0,
        "evaluation_contract": {"diagnostic_schema_strict": False},
        "evaluation_contract_version": "v1",
        "metric_taxonomy_profile_effective": "large_default_v1",
        "diagnostic_schema_ok": True,
        "timing_total_s": 1.23,
        "config_hash": "abc123",
        "git_commit": "deadbee",
        "pid": 12345,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:05+00:00",
        "surrogate_split": {"mode": "kfold", "effective_kfolds": 3},
        "gate_evidence": {
            "selected_stage_evidence": {
                "mandatory_total_metric_count": 3,
                "valid_mandatory_metric_count": 2,
                "invalid_mandatory_metric_count": 1,
                "inactive_mandatory_metric_count": 0,
                "active_invalid_mandatory_metric_count": 1,
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
                "min_valid_mandatory_count_effective": 2,
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
                "pass_rate_optional_case": 0.6,
                "pass_rate_optional_metric_mean": 0.7,
                "mean_rel_diff_mandatory": 0.22,
                "mean_rel_diff_mandatory_all_units": 0.45,
                "mean_rel_diff_mandatory_raw": 0.28,
                "mean_rel_diff_mandatory_family_weighted": 0.22,
                "mean_rel_diff_mandatory_winsorized": 0.20,
                "mandatory_rel_outlier_ratio": 0.1,
                "mandatory_rel_outlier_ratio_all_units": 0.2,
                "mandatory_rel_outlier_ratio_max_effective": 0.2,
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
                "mean_rel_diff_optional": 0.31,
                "mean_rel_diff_all_metric_legacy": 0.9,
                "error_gate_score": 0.75,
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
                "pooling_candidate_scores": [
                    {"backend": "pyg", "coverage_proxy": 0.62},
                    {"backend": "numpy", "coverage_proxy": 0.66},
                ],
                "effective_metric_count": 9,
                "suppressed_low_signal_metric_count": 1,
                "metric_clip_guardrail_trigger_ratio": 0.03,
                "replay_health_trust_invalid": False,
                "structure_feedback_multiplier": 1.10,
                "metric_drift_raw": 1.72,
                "metric_drift_effective": 1.30,
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
                "timing_stage_s": 0.22,
                "timing_pooling_fit_s": 0.02,
                "timing_bridge_s": 0.11,
                "timing_surrogate_eval_s": 0.07,
                "timing_physical_gate_s": 0.04,
                "timing_projection_s": 0.03,
            }
        },
        "primary_blocker_layer": "error",
        "secondary_blockers": ["structure"],
        "validity_fail_reason_primary": "none",
        "selected_metrics": {
            "species_before": 20,
            "species_after": 8,
            "reactions_before": 40,
            "reactions_after": 15,
            "pass_rate": 0.7,
            "mean_rel_diff": 0.25,
            "conservation_violation": 0.0,
            "qoi_metrics_count": 9,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary))

    output_path = tmp_path / "summary_eval.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "cantera_model.cli.summarize_reduction_eval",
            "--report-dir",
            str(reports),
            "--output",
            str(output_path),
            "--entry",
            "baseline:r0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    payload = json.loads(output_path.read_text())
    entry = payload["entries"][0]
    assert entry["mandatory_total_metric_count"] == 3
    assert entry["evaluation_contract_version"] == "v1"
    assert entry["metric_taxonomy_profile_effective"] == "large_default_v1"
    assert entry["diagnostic_schema_ok"] is True
    assert entry["valid_mandatory_metric_count"] == 2
    assert entry["mandatory_total_gate_unit_count"] == 2
    assert entry["valid_mandatory_gate_unit_count"] == 2
    assert entry["valid_mandatory_gate_unit_count_case_rate"] == 1
    assert entry["valid_mandatory_gate_unit_count_coverage"] == 2
    assert entry["mandatory_validity_basis_effective"] == "coverage_evaluable"
    assert entry["active_invalid_mandatory_gate_unit_keys"] == ["A"]
    assert entry["mandatory_gate_unit_evaluable_case_rates"] == {"A": 1.0, "B": 1.0}
    assert entry["mandatory_quality_gate_unit_count"] == 2
    assert entry["mandatory_quality_metric_count"] == 2
    assert entry["mandatory_gate_unit_mode_effective"] == "species_family_quorum"
    assert entry["mandatory_species_family_score_mode_effective"] == "weighted"
    assert entry["mandatory_quality_scope_effective"] == "valid_only"
    assert entry["mandatory_tail_scope_effective"] == "quality_scope"
    assert abs(float(entry["mandatory_species_family_case_pass_min_effective"]) - 0.67) < 1.0e-12
    assert entry["mandatory_validity_passed"] is True
    assert abs(float(entry["pass_rate_mandatory_case"]) - 0.8) < 1.0e-12
    assert abs(float(entry["pass_rate_mandatory_case_all_units"]) - 0.6) < 1.0e-12
    assert abs(float(entry["pass_rate_mandatory_case_all_required"]) - 0.5) < 1.0e-12
    assert abs(float(entry["pass_rate_mandatory_case_ratio_mean"]) - 0.8) < 1.0e-12
    assert abs(float(entry["pass_rate_mandatory_case_all_required_all_units"]) - 0.4) < 1.0e-12
    assert abs(float(entry["pass_rate_mandatory_case_ratio_mean_all_units"]) - 0.6) < 1.0e-12
    assert abs(float(entry["pass_rate_all_metric_legacy"]) - 0.2) < 1.0e-12
    assert entry["mandatory_case_mode_effective"] == "ratio_mean"
    assert entry["mandatory_case_unit_weight_mode_effective"] == "family_weighted"
    assert abs(float(entry["pass_rate_optional_case"]) - 0.6) < 1.0e-12
    assert abs(float(entry["pass_rate_optional_metric_mean"]) - 0.7) < 1.0e-12
    assert abs(float(entry["mean_rel_diff_mandatory"]) - 0.22) < 1.0e-12
    assert abs(float(entry["mean_rel_diff_mandatory_all_units"]) - 0.45) < 1.0e-12
    assert abs(float(entry["mean_rel_diff_mandatory_raw"]) - 0.28) < 1.0e-12
    assert abs(float(entry["mean_rel_diff_mandatory_family_weighted"]) - 0.22) < 1.0e-12
    assert abs(float(entry["mean_rel_diff_mandatory_winsorized"]) - 0.20) < 1.0e-12
    assert abs(float(entry["mandatory_rel_outlier_ratio"]) - 0.1) < 1.0e-12
    assert abs(float(entry["mandatory_rel_outlier_ratio_all_units"]) - 0.2) < 1.0e-12
    assert abs(float(entry["mandatory_rel_outlier_ratio_max_effective"]) - 0.2) < 1.0e-12
    assert abs(float(entry["mandatory_rel_diff_p95"]) - 1.2) < 1.0e-12
    assert abs(float(entry["mandatory_rel_diff_p95_all_units"]) - 1.8) < 1.0e-12
    assert entry["mandatory_tail_guard_passed"] is True
    assert entry["mandatory_tail_guard_triggered"] is True
    assert entry["mandatory_tail_guard_hard_applied"] is False
    assert entry["mandatory_tail_guard_mode_effective"] == "p95"
    assert entry["mandatory_tail_guard_policy_effective"] == "conditional_hard"
    assert abs(float(entry["mandatory_tail_activation_ratio_min_effective"]) - 0.10) < 1.0e-12
    assert entry["mandatory_tail_exceed_ref_effective"] == "tail_max"
    assert abs(float(entry["mandatory_tail_exceed_ratio"]) - 0.05) < 1.0e-12
    assert abs(float(entry["mandatory_tail_rel_diff_max_effective"]) - 1.5) < 1.0e-12
    assert entry["mandatory_quality_scope_empty"] is False
    assert entry["mandatory_mean_aggregation_effective"] == "family_weighted"
    assert entry["mandatory_mean_mode_effective"] == "winsorized"
    assert abs(float(entry["mean_rel_diff_optional"]) - 0.31) < 1.0e-12
    assert abs(float(entry["mean_rel_diff_all_metric_legacy"]) - 0.9) < 1.0e-12
    assert abs(float(entry["error_gate_score"]) - 0.75) < 1.0e-12
    assert entry["coverage_gate_passed"] is True
    assert entry["mandatory_quality_passed"] is True
    assert entry["optional_quality_passed"] is True
    assert entry["mandatory_error_passed"] is True
    assert entry["optional_error_passed"] is True
    assert entry["mandatory_error_include_validity_effective"] is False
    assert entry["error_fail_reason_primary"] == "none"
    assert entry["selection_pool_kind"] == "floor"
    assert abs(float(entry["structure_deficit_score"]) - 0.2) < 1.0e-12
    assert int(entry["pooling_candidate_count"]) == 2
    assert int(entry["pooling_candidate_unique_count"]) == 2
    assert entry["pooling_candidate_selected_backend"] == "pyg"
    assert entry["pooling_candidate_selected_source"] == "swap_refine"
    assert abs(float(entry["pooling_candidate_selected_coverage_proxy"]) - 0.66) < 1.0e-12
    assert abs(float(entry["pooling_candidate_selected_dynamics_recon_error"]) - 0.19) < 1.0e-12
    assert abs(float(entry["pooling_candidate_selected_max_cluster_size_ratio"]) - 0.41) < 1.0e-12
    assert isinstance(entry["pooling_candidate_scores"], list)
    assert len(entry["pooling_candidate_scores"]) == 2
    assert int(entry["effective_metric_count"]) == 9
    assert int(entry["suppressed_low_signal_metric_count"]) == 1
    assert abs(float(entry["metric_clip_guardrail_trigger_ratio"]) - 0.03) < 1.0e-12
    assert entry["replay_health_trust_invalid"] is False
    assert abs(float(entry["structure_feedback_multiplier"]) - 1.10) < 1.0e-12
    assert abs(float(entry["metric_drift_raw"]) - 1.72) < 1.0e-12
    assert abs(float(entry["metric_drift_effective"]) - 1.30) < 1.0e-12
    assert abs(float(entry["metric_drift_effective_cap"]) - 1.30) < 1.0e-12
    assert abs(float(entry["selection_quality_score_raw_drift"]) - 0.64) < 1.0e-12
    assert entry["compression_refine_applied"] is True
    assert int(entry["compression_refine_trials"]) == 2
    assert int(entry["compression_refine_reaction_delta"]) == 3
    assert int(entry["compression_refine_species_delta"]) == 0
    assert entry["compression_refine_mode_effective"] == "baseline_grid"
    assert entry["compression_refine_guard_passed"] is True
    assert int(entry["mandatory_gate_unit_valid_count_shadow_evaluable_ratio"]) == 1
    assert (
        abs(float(entry["mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective"]) - 0.25)
        < 1.0e-12
    )
    assert abs(float(entry["timing_total_s"]) - 1.23) < 1.0e-12
    assert abs(float(entry["timing_stage_s"]) - 0.22) < 1.0e-12
    assert abs(float(entry["timing_bridge_s"]) - 0.11) < 1.0e-12
    assert entry["primary_blocker_layer"] == "error"
    assert entry["secondary_blockers"] == ["structure"]
    assert entry["validity_fail_reason_primary"] == "none"
    assert entry["config_hash"] == "abc123"
    assert entry["git_commit"] == "deadbee"
    assert entry["pid"] == 12345
    assert entry["mode_collapse_warning"] is False
