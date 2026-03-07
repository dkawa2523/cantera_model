from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

LARGE9 = [
    "configs/reduce_ac_benchmark_large_baseline.yaml",
    "configs/reduce_ac_benchmark_large_learnckpp.yaml",
    "configs/reduce_ac_benchmark_large_pooling.yaml",
    "configs/reduce_diamond_benchmarks_large_baseline.yaml",
    "configs/reduce_diamond_benchmarks_large_learnckpp.yaml",
    "configs/reduce_diamond_benchmarks_large_pooling.yaml",
    "configs/reduce_sif4_benchmark_sin3n4_large_baseline.yaml",
    "configs/reduce_sif4_benchmark_sin3n4_large_learnckpp.yaml",
    "configs/reduce_sif4_benchmark_sin3n4_large_pooling.yaml",
]


def _load_cfg(rel: str) -> dict:
    return yaml.safe_load((ROOT / rel).read_text())


def test_large9_have_replay_health_trust_threshold_key() -> None:
    for rel in LARGE9:
        cfg = _load_cfg(rel)
        eval_cfg = dict(cfg.get("evaluation") or {})
        validity = dict(eval_cfg.get("gate_metric_validity") or {})
        assert "min_guardrail_trigger_ratio" in validity, rel


def test_large_learnckpp_have_structure_feedback_keys() -> None:
    for rel in LARGE9:
        if "learnckpp" not in rel:
            continue
        cfg = _load_cfg(rel)
        adp = dict((cfg.get("learnckpp") or {}).get("adaptive_keep_ratio") or {})
        sf = dict(adp.get("structure_feedback") or {})
        assert sf.get("enabled") is True, rel
        assert "alpha_domain" in sf, rel
        assert "beta_balance" in sf, rel
        assert "max_multiplier" in sf, rel


def test_large9_have_tiered_error_aggregation_and_tiers() -> None:
    for rel in LARGE9:
        cfg = _load_cfg(rel)
        eval_cfg = dict(cfg.get("evaluation") or {})
        tiers = dict(eval_cfg.get("gate_metric_tiers") or {})
        mandatory = list(tiers.get("mandatory") or [])
        optional = list(tiers.get("optional") or [])
        assert mandatory, rel
        assert "ignition_delay" in optional, rel
        err = dict(eval_cfg.get("error_aggregation") or {})
        assert str(err.get("mode", "")).lower() == "tiered", rel
        assert "mandatory_case_pass_min" in err, rel
        assert str(err.get("mandatory_case_mode", "")).lower() == "ratio_mean", rel
        assert str(err.get("mandatory_case_unit_weight_mode", "")).lower() in {
            "uniform",
            "family_weighted",
        }, rel
        assert str(err.get("mandatory_quality_scope", "")).lower() in {
            "valid_only",
            "all_units",
            "hybrid_score",
        }, rel
        assert str(err.get("mandatory_tail_scope", "")).lower() in {
            "quality_scope",
            "all_units",
        }, rel
        assert str(err.get("mandatory_mean_mode", "")).lower() == "winsorized", rel
        assert "mandatory_winsor_cap_multiplier" in err, rel
        assert "mandatory_outlier_multiplier" in err, rel
        assert "mandatory_outlier_ratio_max" in err, rel
        assert "optional_metric_pass_min" in err, rel
        norm = dict(eval_cfg.get("metric_normalization") or {})
        assert str(norm.get("denominator_mode", "")).lower() == "max_abs_or_floor", rel
        assert "metric_family_abs_floor" in norm, rel


def test_large9_have_case_rate_mandatory_metric_validity_keys() -> None:
    for rel in LARGE9:
        cfg = _load_cfg(rel)
        eval_cfg = dict(cfg.get("evaluation") or {})
        validity = dict(eval_cfg.get("gate_metric_validity") or {})
        assert str(validity.get("mandatory_metric_validity_mode", "")).lower() == "case_pass_rate", rel
        assert "mandatory_metric_case_pass_min" in validity, rel
        assert str(validity.get("mandatory_validity_basis", "")).lower() == "coverage_evaluable", rel


def test_large9_mode_trios_use_same_split_mode() -> None:
    groups = {
        "ac": [
            "configs/reduce_ac_benchmark_large_baseline.yaml",
            "configs/reduce_ac_benchmark_large_learnckpp.yaml",
            "configs/reduce_ac_benchmark_large_pooling.yaml",
        ],
        "diamond": [
            "configs/reduce_diamond_benchmarks_large_baseline.yaml",
            "configs/reduce_diamond_benchmarks_large_learnckpp.yaml",
            "configs/reduce_diamond_benchmarks_large_pooling.yaml",
        ],
        "sif4": [
            "configs/reduce_sif4_benchmark_sin3n4_large_baseline.yaml",
            "configs/reduce_sif4_benchmark_sin3n4_large_learnckpp.yaml",
            "configs/reduce_sif4_benchmark_sin3n4_large_pooling.yaml",
        ],
    }
    for _, rels in groups.items():
        split_modes = set()
        for rel in rels:
            cfg = _load_cfg(rel)
            eval_cfg = dict(cfg.get("evaluation") or {})
            split = dict(eval_cfg.get("surrogate_split") or {})
            split_modes.add(str(split.get("mode", "")).lower())
        assert len(split_modes) == 1, f"split mode mismatch in {rels}: {split_modes}"


def test_large9_have_v33_quorum_and_tail_guard_keys() -> None:
    for rel in LARGE9:
        cfg = _load_cfg(rel)
        eval_cfg = dict(cfg.get("evaluation") or {})
        validity = dict(eval_cfg.get("gate_metric_validity") or {})
        assert str(validity.get("mandatory_valid_unit_mode", "")).lower() == "species_family_quorum", rel
        assert str(validity.get("mandatory_species_family_score_mode", "")).lower() == "weighted", rel
        assert float(validity.get("mandatory_species_family_case_pass_min", 0.0)) > 0.0, rel
        err = dict(eval_cfg.get("error_aggregation") or {})
        assert str(err.get("mandatory_tail_guard_mode", "")).lower() == "p95", rel
        assert float(err.get("mandatory_tail_rel_diff_max", 0.0)) > 0.0, rel
        assert int(err.get("mandatory_tail_min_samples", 0)) >= 1, rel


def test_large9_have_v34_orthogonalized_error_keys() -> None:
    for rel in LARGE9:
        cfg = _load_cfg(rel)
        eval_cfg = dict(cfg.get("evaluation") or {})
        validity = dict(eval_cfg.get("gate_metric_validity") or {})
        assert str(validity.get("validity_scope", "")).lower() == "coverage_only", rel
        err = dict(eval_cfg.get("error_aggregation") or {})
        assert bool(err.get("require_explicit_thresholds")) is True, rel
        assert bool(err.get("mandatory_error_include_validity")) is False, rel
        assert str(err.get("mandatory_tail_guard_policy", "")).lower() == "conditional_hard", rel
        assert float(err.get("mandatory_tail_activation_ratio_min", 0.0)) > 0.0, rel
        assert str(err.get("mandatory_tail_exceed_ref", "")).lower() in {"tail_max", "mean_cap"}, rel


def test_large9_adaptive_kfold_requires_explicit_policy_and_policy_keys() -> None:
    for rel in LARGE9:
        cfg = _load_cfg(rel)
        eval_cfg = dict(cfg.get("evaluation") or {})
        split = dict(eval_cfg.get("surrogate_split") or {})
        assert str(split.get("mode", "")).lower() == "adaptive_kfold", rel
        assert bool(split.get("enforce_explicit_policy")) is True, rel
        policy = dict(split.get("kfold_policy") or {})
        assert "min_cases_for_kfold" in policy, rel
        assert "default_k" in policy, rel
        assert isinstance(policy.get("k_by_case_count"), dict), rel


def test_large9_selection_has_nonpass_priority_key() -> None:
    for rel in LARGE9:
        cfg = _load_cfg(rel)
        selection = dict(cfg.get("selection") or {})
        assert str(selection.get("nonpass_priority", "")).lower() in {
            "structure_then_score",
            "score_only",
        }, rel


def test_large9_have_v352_shadow_and_surrogate_drift_keys() -> None:
    for rel in LARGE9:
        cfg = _load_cfg(rel)
        eval_cfg = dict(cfg.get("evaluation") or {})
        validity = dict(eval_cfg.get("gate_metric_validity") or {})
        assert "mandatory_gate_unit_min_evaluable_case_ratio_shadow" in validity, rel
        assert float(validity.get("mandatory_gate_unit_min_evaluable_case_ratio_shadow", -1.0)) >= 0.0, rel

        drift = dict(eval_cfg.get("surrogate_drift") or {})
        assert bool(drift.get("selection_use_raw_drift")) is True, rel
        assert float(drift.get("raw_drift_cap_for_selection", 0.0)) >= 1.0, rel
        assert float(drift.get("keep_effective_drift_cap_for_eval", 0.0)) >= 1.0, rel


def test_large9_have_v36_contract_taxonomy_non_regression_keys() -> None:
    for rel in LARGE9:
        cfg = _load_cfg(rel)
        eval_cfg = dict(cfg.get("evaluation") or {})
        contract = dict(eval_cfg.get("contract") or {})
        assert str(contract.get("version", "")).strip().lower() == "v1", rel
        assert bool(contract.get("enforce")) is True, rel
        assert bool(contract.get("diagnostic_schema_strict")) is True, rel
        assert str(contract.get("evaluation_profile", "")).strip().lower() == "tiered_v35", rel
        assert str(contract.get("run_policy_profile", "")).strip().lower() == "adaptive_kfold_strict_v1", rel

        taxonomy = dict(eval_cfg.get("metric_taxonomy") or {})
        assert str(taxonomy.get("source", "")).strip().lower() == "shared_yaml", rel
        assert str(taxonomy.get("profile", "")).strip().lower() == "large_default_v1", rel
        assert str(taxonomy.get("path", "")).strip() == "configs/evaluation/metric_taxonomy_profiles.yaml", rel

        non_reg = dict(eval_cfg.get("non_regression") or {})
        assert bool(non_reg.get("reaction_reduction_priority")) is True, rel
        assert float(non_reg.get("max_reaction_reduction_drop_ratio", -1.0)) == 0.02, rel


def test_large9_have_v37_compression_optimizer_keys() -> None:
    for rel in LARGE9:
        cfg = _load_cfg(rel)
        eval_cfg = dict(cfg.get("evaluation") or {})
        optimizer = dict(eval_cfg.get("compression_optimizer") or {})
        assert bool(optimizer.get("enabled")) is True, rel
        assert str(optimizer.get("mode", "")).strip().lower() == "deterministic_grid", rel
        assert int(optimizer.get("per_stage_extra_trials", 0)) == 2, rel
        assert bool(optimizer.get("reaction_priority")) is True, rel
        assert float(optimizer.get("max_allowed_mandatory_mean_delta", -1.0)) >= 0.0, rel
        assert float(optimizer.get("max_allowed_optional_mean_delta", -1.0)) >= 0.0, rel
        assert float(optimizer.get("max_allowed_mandatory_pass_rate_drop", -1.0)) >= 0.0, rel
        assert bool(optimizer.get("require_gate_passed")) is True, rel
        assert bool(optimizer.get("require_structure_passed")) is True, rel


def test_large_learnckpp_pooling_have_v37_post_prune_refine_keys() -> None:
    for rel in LARGE9:
        if "baseline" in rel:
            continue
        cfg = _load_cfg(rel)
        learnckpp = dict(cfg.get("learnckpp") or {})
        select_cfg = dict(learnckpp.get("select") or {})
        refine = dict(select_cfg.get("post_prune_refine") or {})
        assert bool(refine.get("enabled")) is True, rel
        assert int(refine.get("max_steps", 0)) >= 1, rel
        assert float(refine.get("importance_quantile_step", 0.0)) > 0.0, rel
        assert bool(refine.get("respect_coverage_postselect")) is True, rel
        assert bool(refine.get("respect_active_cluster_coverage")) is True, rel


def test_large_pooling_have_v381_candidate_selection_keys() -> None:
    for rel in LARGE9:
        if "pooling" not in rel:
            continue
        cfg = _load_cfg(rel)
        pooling = dict(cfg.get("pooling") or {})
        model = dict(pooling.get("model") or {})
        candidates = list(model.get("backend_candidates") or [])
        assert len(candidates) >= 2, rel
        assert len({str(x).strip().lower() for x in candidates if str(x).strip()}) >= 2, rel
        selection = dict(pooling.get("candidate_selection") or {})
        assert bool(selection.get("enabled")) is True, rel
        assert str(selection.get("pick_policy", "")).strip().lower() == "proxy_lexicographic", rel
        assert bool(selection.get("dedupe_by_assignment_hash")) is True, rel
        swap = dict(selection.get("fallback_swap_refine") or {})
        assert bool(swap.get("enabled")) is True, rel
        assert int(swap.get("max_steps", 0)) >= 1, rel
        assert float(swap.get("min_coverage_improve", 0.0)) > 0.0, rel
