from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from cantera_model.eval.diagnostic_schema import SUMMARY_EXPORT_KEYS


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    extra: list[str] = []
    seen = set(keys)
    for row in rows[1:]:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            extra.append(key)
    keys.extend(extra)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _objective_tuple(row: dict[str, Any]) -> tuple[float, float, float]:
    try:
        species = float(row.get("species_after", float("inf")))
    except (TypeError, ValueError):
        species = float("inf")
    try:
        reactions = float(row.get("reactions_after", float("inf")))
    except (TypeError, ValueError):
        reactions = float("inf")
    try:
        err = float(row.get("mean_rel_diff", float("inf")))
    except (TypeError, ValueError):
        err = float("inf")
    return species, reactions, err


def _dominates(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    return (a[0] <= b[0] and a[1] <= b[1] and a[2] <= b[2]) and (a != b)


def _is_gate_floor_passed(row: dict[str, Any]) -> bool:
    gate_passed = bool(row.get("gate_passed", True))
    floor_passed = bool(row.get("floor_passed", True))
    balance_passed = bool(row.get("balance_gate_passed", True))
    cluster_guard_passed = bool(row.get("cluster_guard_passed", True))
    return gate_passed and floor_passed and balance_passed and cluster_guard_passed


def _pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    filtered = [r for r in rows if _is_gate_floor_passed(r)]
    if not filtered:
        return []
    objs = [_objective_tuple(r) for r in filtered]
    out: list[dict[str, Any]] = []
    for i, row in enumerate(filtered):
        dominated = False
        for j, other in enumerate(filtered):
            if i == j:
                continue
            if _dominates(objs[j], objs[i]):
                dominated = True
                break
        if not dominated:
            out.append(row)
    return out


def write_report(
    report_dir: str | Path,
    *,
    run_id: str,
    stage_rows: list[dict[str, Any]],
    selected_stage: str | None,
    summary_payload: dict[str, Any],
) -> Path:
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = dict(summary_payload)
    summary["run_id"] = run_id
    summary["selected_stage"] = selected_stage
    selected_metrics = dict(summary.get("selected_metrics") or {})
    gate_evidence = dict(summary.get("gate_evidence") or {})
    selected_stage_evidence = dict(gate_evidence.get("selected_stage_evidence") or {})
    reduction_trace = dict(summary.get("reduction_trace") or {})
    trend = list(reduction_trace.get("candidate_trend") or [])
    selected_row = {}
    if selected_stage is not None:
        for row in trend:
            if str(row.get("stage")) == str(selected_stage):
                selected_row = dict(row)
                break

    def _selected_value(key: str, default: Any) -> Any:
        value = selected_metrics.get(key)
        if value is not None and value != "":
            return value
        value = selected_stage_evidence.get(key)
        if value is not None and value != "":
            return value
        value = selected_row.get(key)
        if value is not None and value != "":
            return value
        return default

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    lines = [
        f"# Reduction Report: {run_id}",
        "",
        f"- selected_stage: {selected_stage}",
        f"- hard_ban_violations: {summary.get('hard_ban_violations', 'n/a')}",
        f"- gate_passed: {summary.get('gate_passed', False)}",
        f"- primary_blocker_layer: {summary.get('primary_blocker_layer', 'none')}",
        f"- validity_fail_reason_primary: {summary.get('validity_fail_reason_primary', 'none')}",
        f"- error_fail_reason_primary: {summary.get('error_fail_reason_primary', 'none')}",
        f"- diagnostic_schema_ok: {summary.get('diagnostic_schema_ok', True)}",
        f"- evaluation_contract_version: {summary.get('evaluation_contract_version', '')}",
        f"- metric_taxonomy_profile_effective: {summary.get('metric_taxonomy_profile_effective', 'legacy_builtin')}",
        f"- timing_total_s: {float(summary.get('timing_total_s', 0.0)):.4f}",
        f"- diagnostic_export_key_count: {len(SUMMARY_EXPORT_KEYS)}",
        f"- compression_refine_applied: {bool(_selected_value('compression_refine_applied', False))}",
        f"- compression_refine_trials: {int(_selected_value('compression_refine_trials', 0) or 0)}",
        f"- compression_refine_reaction_delta: {int(_selected_value('compression_refine_reaction_delta', 0) or 0)}",
        f"- compression_refine_mode_effective: {_selected_value('compression_refine_mode_effective', 'none')}",
        f"- compression_refine_guard_passed: {bool(_selected_value('compression_refine_guard_passed', True))}",
        f"- pooling_candidate_count: {int(_selected_value('pooling_candidate_count', 0) or 0)}",
        f"- pooling_candidate_unique_count: {int(_selected_value('pooling_candidate_unique_count', 0) or 0)}",
        f"- pooling_candidate_selected_backend: {_selected_value('pooling_candidate_selected_backend', '')}",
        f"- pooling_candidate_selected_source: {_selected_value('pooling_candidate_selected_source', 'backend')}",
        f"- pooling_candidate_selected_coverage_proxy: {float(_selected_value('pooling_candidate_selected_coverage_proxy', 0.0) or 0.0):.6f}",
        f"- pooling_candidate_selected_dynamics_recon_error: {float(_selected_value('pooling_candidate_selected_dynamics_recon_error', 0.0) or 0.0):.6f}",
        f"- pooling_candidate_selected_max_cluster_size_ratio: {float(_selected_value('pooling_candidate_selected_max_cluster_size_ratio', 0.0) or 0.0):.6f}",
    ]

    if trend:
        lines.extend(
            [
                "",
                "## Candidate Trend",
                "",
                "| stage | species_after | reactions_after | overall_candidates | overall_selected | select_ratio | mean_rel_diff | floor_passed | balance_gate_passed | cluster_guard | weighted_cov | top_weighted_cov | essential_cov | max_cluster_ratio | balance_mode | selection_score | rs_upper_eff | active_cov_floor_eff | balance_margin | dynamic | valid_mandatory | mandatory_total | inactive_mandatory | active_invalid_mandatory | mandatory_validity | cov_gate | mand_quality | opt_quality | mand_error | opt_error | mand_case_pass | mand_case_all | mand_case_ratio | mand_case_mode | opt_case_pass | opt_metric_pass | mean_rel_mand_raw | mean_rel_mand_fam | mean_rel_mand_win | mand_outlier_ratio | mand_mean_mode | err_gate_score | eff_metrics | suppressed_low_signal | clip_guardrail_ratio | replay_trust_invalid | structure_fb_mult | drift_raw | drift_cap | sel_quality_raw_drift | valid_shadow_evaluable | shadow_ratio_min | blocker | validity_reason | error_reason | tail_trigger | tail_hard | tail_policy | tail_exceed_ratio | t_stage_s | t_bridge_s | t_surr_s | t_phys_s | t_proj_s |",
                "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|---:|---:|---:|---:|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|:---:|:---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in trend:
            lines.append(
                "| {stage} | {species_after} | {reactions_after} | {overall_candidates} | {overall_selected} | {overall_select_ratio:.4f} | {mean_rel_diff:.4f} | {floor_passed} | {balance_gate_passed} | {cluster_guard_passed} | {weighted_cov:.4f} | {top_weighted_cov:.4f} | {essential_cov:.4f} | {max_cluster_ratio:.4f} | {balance_mode} | {selection_score:.4f} | {rs_upper_effective:.4f} | {active_cov_effective_floor:.4f} | {balance_margin:.4f} | {dynamic_applied} | {valid_mandatory_metric_count} | {mandatory_total_metric_count} | {inactive_mandatory_metric_count} | {active_invalid_mandatory_metric_count} | {mandatory_validity_passed} | {coverage_gate_passed} | {mandatory_quality_passed} | {optional_quality_passed} | {mandatory_error_passed} | {optional_error_passed} | {pass_rate_mandatory_case:.4f} | {pass_rate_mandatory_case_all_required:.4f} | {pass_rate_mandatory_case_ratio_mean:.4f} | {mandatory_case_mode_effective} | {pass_rate_optional_case:.4f} | {pass_rate_optional_metric_mean:.4f} | {mean_rel_diff_mandatory_raw:.4f} | {mean_rel_diff_mandatory_family_weighted:.4f} | {mean_rel_diff_mandatory_winsorized:.4f} | {mandatory_rel_outlier_ratio:.4f} | {mandatory_mean_mode_effective} | {error_gate_score:.4f} | {effective_metric_count} | {suppressed_low_signal_metric_count} | {metric_clip_guardrail_trigger_ratio:.4f} | {replay_health_trust_invalid} | {structure_feedback_multiplier:.4f} | {metric_drift_raw:.4f} | {metric_drift_effective_cap:.4f} | {selection_quality_score_raw_drift:.4f} | {mandatory_gate_unit_valid_count_shadow_evaluable_ratio} | {mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective:.2f} | {primary_blocker_layer} | {validity_fail_reason_primary} | {error_fail_reason_primary} | {mandatory_tail_guard_triggered} | {mandatory_tail_guard_hard_applied} | {mandatory_tail_guard_policy_effective} | {mandatory_tail_exceed_ratio:.4f} | {timing_stage_s:.4f} | {timing_bridge_s:.4f} | {timing_surrogate_eval_s:.4f} | {timing_physical_gate_s:.4f} | {timing_projection_s:.4f} |".format(
                    stage=str(row.get("stage", "")),
                    species_after=int(row.get("species_after", 0)),
                    reactions_after=int(row.get("reactions_after", 0)),
                    overall_candidates=int(row.get("overall_candidates", 0)),
                    overall_selected=int(row.get("overall_selected", 0)),
                    overall_select_ratio=float(row.get("overall_select_ratio", 0.0)),
                    mean_rel_diff=float(row.get("mean_rel_diff", 0.0)),
                    floor_passed=("yes" if bool(row.get("floor_passed", True)) else "no"),
                    balance_gate_passed=("yes" if bool(row.get("balance_gate_passed", True)) else "no"),
                    cluster_guard_passed=("yes" if bool(row.get("cluster_guard_passed", True)) else "no"),
                    weighted_cov=float(row.get("weighted_active_species_coverage", 0.0)),
                    top_weighted_cov=float(
                        row.get("active_species_coverage_top_weighted", row.get("weighted_active_species_coverage", 0.0))
                    ),
                    essential_cov=float(row.get("essential_species_coverage", 1.0)),
                    max_cluster_ratio=float(row.get("max_cluster_size_ratio", 0.0)),
                    balance_mode=str(row.get("balance_mode", "binary")),
                    selection_score=float(row.get("selection_score", 0.0)),
                    rs_upper_effective=float(row.get("rs_upper_effective", 0.0)),
                    active_cov_effective_floor=float(row.get("active_cov_effective_floor", 0.0)),
                    balance_margin=float(row.get("balance_margin", 0.0)),
                    dynamic_applied=("yes" if bool(row.get("balance_dynamic_applied", False)) else "no"),
                    valid_mandatory_metric_count=int(row.get("valid_mandatory_metric_count", 0)),
                    mandatory_total_metric_count=int(row.get("mandatory_total_metric_count", 0)),
                    inactive_mandatory_metric_count=int(row.get("inactive_mandatory_metric_count", 0)),
                    active_invalid_mandatory_metric_count=int(row.get("active_invalid_mandatory_metric_count", 0)),
                    mandatory_validity_passed=("yes" if bool(row.get("mandatory_validity_passed", True)) else "no"),
                    coverage_gate_passed=("yes" if bool(row.get("coverage_gate_passed", True)) else "no"),
                    mandatory_quality_passed=("yes" if bool(row.get("mandatory_quality_passed", True)) else "no"),
                    optional_quality_passed=("yes" if bool(row.get("optional_quality_passed", True)) else "no"),
                    mandatory_error_passed=("yes" if bool(row.get("mandatory_error_passed", True)) else "no"),
                    optional_error_passed=("yes" if bool(row.get("optional_error_passed", True)) else "no"),
                    pass_rate_mandatory_case=float(row.get("pass_rate_mandatory_case", row.get("pass_rate", 0.0))),
                    pass_rate_mandatory_case_all_required=float(
                        row.get(
                            "pass_rate_mandatory_case_all_required",
                            row.get("pass_rate_mandatory_case", row.get("pass_rate", 0.0)),
                        )
                    ),
                    pass_rate_mandatory_case_ratio_mean=float(
                        row.get(
                            "pass_rate_mandatory_case_ratio_mean",
                            row.get("pass_rate_mandatory_case", row.get("pass_rate", 0.0)),
                        )
                    ),
                    mandatory_case_mode_effective=str(row.get("mandatory_case_mode_effective", "ratio_mean")),
                    pass_rate_optional_case=float(row.get("pass_rate_optional_case", row.get("pass_rate", 0.0))),
                    pass_rate_optional_metric_mean=float(
                        row.get("pass_rate_optional_metric_mean", row.get("pass_rate", 0.0))
                    ),
                    mean_rel_diff_mandatory_raw=float(
                        row.get("mean_rel_diff_mandatory_raw", row.get("mean_rel_diff_mandatory", row.get("mean_rel_diff", 0.0)))
                    ),
                    mean_rel_diff_mandatory_family_weighted=float(
                        row.get(
                            "mean_rel_diff_mandatory_family_weighted",
                            row.get("mean_rel_diff_mandatory", row.get("mean_rel_diff", 0.0)),
                        )
                    ),
                    mean_rel_diff_mandatory_winsorized=float(
                        row.get(
                            "mean_rel_diff_mandatory_winsorized",
                            row.get("mean_rel_diff_mandatory", row.get("mean_rel_diff", 0.0)),
                        )
                    ),
                    mandatory_rel_outlier_ratio=float(row.get("mandatory_rel_outlier_ratio", 0.0)),
                    mandatory_mean_mode_effective=str(row.get("mandatory_mean_mode_effective", "winsorized")),
                    error_gate_score=float(row.get("error_gate_score", 0.0)),
                    effective_metric_count=int(row.get("effective_metric_count", row.get("qoi_metrics_count", 0))),
                    suppressed_low_signal_metric_count=int(row.get("suppressed_low_signal_metric_count", 0)),
                    metric_clip_guardrail_trigger_ratio=float(row.get("metric_clip_guardrail_trigger_ratio", 0.0)),
                    replay_health_trust_invalid=("yes" if bool(row.get("replay_health_trust_invalid", False)) else "no"),
                    structure_feedback_multiplier=float(row.get("structure_feedback_multiplier", 1.0)),
                    metric_drift_raw=float(row.get("metric_drift_raw", row.get("metric_drift_effective", 1.0))),
                    metric_drift_effective_cap=float(row.get("metric_drift_effective_cap", 1.30)),
                    selection_quality_score_raw_drift=float(row.get("selection_quality_score_raw_drift", 0.0)),
                    mandatory_gate_unit_valid_count_shadow_evaluable_ratio=int(
                        row.get("mandatory_gate_unit_valid_count_shadow_evaluable_ratio", 0)
                    ),
                    mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective=float(
                        row.get("mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective", 0.25)
                    ),
                    primary_blocker_layer=str(row.get("primary_blocker_layer", "none")),
                    validity_fail_reason_primary=str(row.get("validity_fail_reason_primary", "none")),
                    error_fail_reason_primary=str(row.get("error_fail_reason_primary", "none")),
                    mandatory_tail_guard_triggered=("yes" if bool(row.get("mandatory_tail_guard_triggered", False)) else "no"),
                    mandatory_tail_guard_hard_applied=("yes" if bool(row.get("mandatory_tail_guard_hard_applied", False)) else "no"),
                    mandatory_tail_guard_policy_effective=str(row.get("mandatory_tail_guard_policy_effective", "conditional_hard")),
                    mandatory_tail_exceed_ratio=float(row.get("mandatory_tail_exceed_ratio", 0.0)),
                    timing_stage_s=float(row.get("timing_stage_s", 0.0)),
                    timing_bridge_s=float(row.get("timing_bridge_s", 0.0)),
                    timing_surrogate_eval_s=float(row.get("timing_surrogate_eval_s", 0.0)),
                    timing_physical_gate_s=float(row.get("timing_physical_gate_s", 0.0)),
                    timing_projection_s=float(row.get("timing_projection_s", 0.0)),
                )
            )

    cluster_preview = dict(reduction_trace.get("cluster_preview") or {})
    selected_clusters = list(cluster_preview.get("selected_stage_clusters") or [])
    if selected_clusters:
        lines.extend(
            [
                "",
                "## Selected Stage Clusters",
                "",
                "| cluster_id | size | members_sample | elements |",
                "|---:|---:|---|---|",
            ]
        )
        for row in selected_clusters:
            members = ", ".join(str(x) for x in list(row.get("members_sample") or []))
            elems = ", ".join(str(x) for x in list(row.get("elements") or []))
            lines.append(
                f"| {int(row.get('cluster_id', 0))} | {int(row.get('size', 0))} | {members} | {elems} |"
            )

    (out_dir / "report.md").write_text("\n".join(lines))

    _write_csv(out_dir / "metrics.csv", stage_rows)
    _write_csv(out_dir / "pareto.csv", _pareto_front(stage_rows))
    return out_dir
