from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cantera_model.eval.cantera_runner import load_conditions
from cantera_model.eval.surrogate_eval import compare_with_baseline, fit_lightweight_surrogate, run_surrogate_cases
from cantera_model.io.trace_store import load_case_bundle
from cantera_model.network.flux import build_flux, reaction_importance
from cantera_model.network.stoich import build_nu, extract_species_meta
from cantera_model.reporting.report import write_report
from cantera_model.reduction.conservation import conservation_violation, project_to_conservation
from cantera_model.reduction.learnckpp.candidate_reactions import generate_overall_candidates
from cantera_model.reduction.learnckpp.rate_model import fit_rate_model, predict_rates
from cantera_model.reduction.learnckpp.simulate import simulate_reduced
from cantera_model.reduction.learnckpp.sparse_select import select_sparse_overall
from cantera_model.reduction.pooling.constraints import build_hard_mask, build_pairwise_cost
from cantera_model.reduction.pooling.export import save_pooling_artifact
from cantera_model.reduction.pooling.features import extract_species_features
from cantera_model.reduction.pooling.graphs import build_bipartite_graph, build_species_graph
from cantera_model.reduction.pooling.train import train_pooling_assignment
from cantera_model.reduction.merge_free import DEFAULT_POLICY, fit_merge_mapping
from cantera_model.reduction.prune_gate import train_prune_gate
from cantera_model.types import ReductionMapping


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    return data


def _resolve_path(raw: str | Path, *, base: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (base / path).resolve()


def _default_species_meta() -> list[dict[str, Any]]:
    return [
        {"name": "CH4", "composition": {"C": 1, "H": 4}, "phase": "gas", "charge": 0, "radical": False, "role": "fuel"},
        {"name": "CH", "composition": {"C": 1, "H": 1}, "phase": "gas", "charge": 0, "radical": True, "role": "fuel"},
        {"name": "CF4", "composition": {"C": 1, "F": 4}, "phase": "gas", "charge": 0, "radical": False, "role": "etch"},
        {"name": "F", "composition": {"F": 1}, "phase": "gas", "charge": 0, "radical": True, "role": "etch"},
        {"name": "N", "composition": {"N": 1}, "phase": "gas", "charge": 0, "radical": True, "role": "carrier"},
        {"name": "H", "composition": {"H": 1}, "phase": "gas", "charge": 0, "radical": True, "role": "carrier"},
    ]


def _stage_defaults() -> list[dict[str, Any]]:
    return [
        {"name": "A", "target_ratio": 0.75, "penalty_scale": 1.0, "prune_lambda": 0.001, "prune_keep_ratio": 0.9, "prune_threshold": 0.45, "prune_exact_keep": True, "metric_drift": 1.02},
        {"name": "B", "target_ratio": 0.55, "penalty_scale": 0.7, "prune_lambda": 0.004, "prune_keep_ratio": 0.7, "prune_threshold": 0.40, "prune_exact_keep": True, "metric_drift": 1.08},
        {"name": "C", "target_ratio": 0.35, "penalty_scale": 0.4, "prune_lambda": 0.01, "prune_keep_ratio": 0.5, "prune_threshold": 0.35, "prune_exact_keep": True, "metric_drift": 1.15},
    ]


def _build_element_matrix(species_meta: list[dict[str, Any]]) -> np.ndarray:
    all_elems = sorted({e for sp in species_meta for e in (sp.get("composition") or {}).keys()})
    if not all_elems:
        return np.eye(len(species_meta), dtype=float)
    A = np.zeros((len(all_elems), len(species_meta)), dtype=float)
    for j, sp in enumerate(species_meta):
        comp = sp.get("composition") or {}
        for i, elem in enumerate(all_elems):
            A[i, j] = float(comp.get(elem, 0.0))
    return A


def _is_gas_phase_name(phase: Any) -> bool:
    text = str(phase or "").strip().lower()
    return text in {"", "gas", "gas_phase", "ideal_gas"}


def _reaction_domain_masks(nu: np.ndarray, species_meta: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    nu_arr = np.asarray(nu, dtype=float)
    if nu_arr.ndim != 2:
        raise ValueError("nu must be 2-D")
    n_species, n_reactions = nu_arr.shape
    if len(species_meta) != n_species:
        gas_mask = np.ones((n_reactions,), dtype=bool)
        surf_mask = np.zeros((n_reactions,), dtype=bool)
        return gas_mask, surf_mask
    non_gas_species = np.asarray([not _is_gas_phase_name(m.get("phase")) for m in species_meta], dtype=bool)
    if not np.any(non_gas_species):
        gas_mask = np.ones((n_reactions,), dtype=bool)
        surf_mask = np.zeros((n_reactions,), dtype=bool)
        return gas_mask, surf_mask
    surface_reaction_mask = np.any(np.abs(nu_arr[non_gas_species, :]) > 1.0e-14, axis=0)
    gas_reaction_mask = ~surface_reaction_mask
    return np.asarray(gas_reaction_mask, dtype=bool), np.asarray(surface_reaction_mask, dtype=bool)


def _cluster_domain_counts(S: np.ndarray, species_meta: list[dict[str, Any]]) -> dict[str, int]:
    s = np.asarray(S, dtype=float)
    if s.ndim != 2:
        return {"gas": 0, "surface": 0}
    n_species = s.shape[0]
    if len(species_meta) != n_species:
        return {"gas": int(s.shape[1]), "surface": 0}
    gas_species = np.asarray([_is_gas_phase_name(m.get("phase")) for m in species_meta], dtype=bool)
    gas_clusters = 0
    surface_clusters = 0
    for c in range(s.shape[1]):
        members = np.where(s[:, c] > 0.5)[0]
        if members.size == 0:
            continue
        if bool(np.all(gas_species[members])):
            gas_clusters += 1
        else:
            surface_clusters += 1
    return {"gas": int(gas_clusters), "surface": int(surface_clusters)}


def _make_synthetic_dynamics(n_species: int, n_reactions: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    nu = rng.integers(-2, 3, size=(n_species, n_reactions)).astype(float)

    for j in range(n_reactions):
        if np.all(nu[:, j] >= 0) or np.all(nu[:, j] <= 0):
            nu[rng.integers(0, n_species), j] = -1.0
            nu[rng.integers(0, n_species), j] = 1.0

    # Keep synthetic dynamics numerically stable for physical gate checks.
    nu = np.asarray(nu, dtype=float) * 1.0e-3
    rop = np.abs(rng.normal(loc=1.0, scale=0.4, size=(120, n_reactions)))
    wdot = rop @ nu.T
    return nu, rop, wdot


def _merge_policy_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    merge_cfg = cfg.get("merge") or {}
    physics_cfg = cfg.get("physics_constraints") or {}
    policy = dict(DEFAULT_POLICY)

    hard = dict((policy.get("hard") or {}))
    hard.update((merge_cfg.get("hard") or {}))
    if "phase_mixing_forbidden" in physics_cfg:
        hard["phase_mixing_forbidden"] = bool(physics_cfg.get("phase_mixing_forbidden"))
    if "surface_site_family_strict" in physics_cfg:
        hard["surface_site_family_strict"] = bool(physics_cfg.get("surface_site_family_strict"))
    policy["hard"] = hard

    soft_penalty = dict(((policy.get("soft") or {}).get("penalty") or {}))
    soft_penalty.update(((merge_cfg.get("soft") or {}).get("penalty") or {}))
    soft = dict((policy.get("soft") or {}))
    soft["penalty"] = soft_penalty
    if "lumpability_weight" in (merge_cfg.get("soft") or {}):
        soft["lumpability_weight"] = float((merge_cfg.get("soft") or {}).get("lumpability_weight", 0.0))
    policy["soft"] = soft

    weights = dict(policy.get("weights") or {})
    weights.update((merge_cfg.get("weights") or {}))
    policy["weights"] = weights

    policy["overlap_method"] = str(merge_cfg.get("overlap_method", policy.get("overlap_method", "jaccard")))
    search = dict(policy.get("search") or {})
    search.update((cfg.get("search") or {}).get("merge_search") or {})
    policy["search"] = search
    return policy


def _choose_stage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [r for r in rows if r.get("gate_passed")]
    passed_non_degraded = [r for r in passed if not bool(r.get("physical_degraded", False))]
    non_degraded = [r for r in rows if not bool(r.get("physical_degraded", False))]

    if passed_non_degraded:
        passed_non_degraded.sort(key=lambda r: (r["species_after"], r["reactions_after"], r["mean_rel_diff"]))
        return passed_non_degraded[0]
    if passed:
        passed.sort(key=lambda r: (r["species_after"], r["reactions_after"], r["mean_rel_diff"]))
        return passed[0]
    if non_degraded:
        non_degraded.sort(key=lambda r: (r["mean_rel_diff"], r["species_after"], r["reactions_after"]))
        return non_degraded[0]
    rows.sort(key=lambda r: (r["mean_rel_diff"], r["species_after"], r["reactions_after"]))
    return rows[0]


def _resolve_stage_scheduled_value(
    raw: Any,
    *,
    stage_idx: int,
    stage_name: str,
    default: Any,
) -> Any:
    if raw is None:
        return default
    if isinstance(raw, (list, tuple)):
        if not raw:
            return default
        idx = min(max(int(stage_idx), 0), len(raw) - 1)
        return raw[idx]
    if isinstance(raw, dict):
        for key in (stage_name, str(stage_idx), str(stage_idx + 1), "default"):
            if key in raw:
                return raw[key]
        return default
    return raw


def _resolve_physics_profile(
    cfg: dict[str, Any],
    trace_meta: dict[str, Any] | None = None,
    mech_path: str | None = None,
) -> dict[str, Any]:
    explicit = cfg.get("physics_profile")
    if explicit:
        profile_name = str(explicit).strip()
        source = "config"
    else:
        text_parts = [
            str(cfg.get("run_id", "")),
            str(cfg.get("trace_h5", "")),
            str(cfg.get("network_dir", "")),
            str(cfg.get("conditions_csv", "")),
            str(mech_path or ""),
        ]
        if trace_meta:
            text_parts.extend(
                [
                    str(trace_meta.get("source", "")),
                    str(trace_meta.get("trace_h5", "")),
                    " ".join(str(x) for x in list(trace_meta.get("species_names") or [])[:20]),
                ]
            )
        text = " ".join(text_parts).lower()
        if "sif4" in text or "sin3n4" in text:
            profile_name = "benchmark_sif4_sin3n4_cvd"
        elif "benchmark_large" in text or "ac_hydrocarbon" in text or "ac_benchmark" in text:
            profile_name = "benchmark_large"
        elif "diamond" in text:
            profile_name = "benchmarks_diamond"
        else:
            profile_name = "default"
        source = "inferred"

    floor_defaults = {
        "default": {"min_species_abs": 0, "min_species_ratio": 0.0, "min_reactions_abs": 0, "min_reactions_ratio": 0.0},
        "benchmarks_diamond": {"min_species_abs": 3, "min_species_ratio": 0.20, "min_reactions_abs": 2, "min_reactions_ratio": 0.10},
        "benchmark_sif4_sin3n4_cvd": {"min_species_abs": 8, "min_species_ratio": 0.08, "min_reactions_abs": 10, "min_reactions_ratio": 0.03},
        "benchmark_large": {"min_species_abs": 8, "min_species_ratio": 0.08, "min_reactions_abs": 12, "min_reactions_ratio": 0.03},
    }
    cfg_floors = dict(cfg.get("physics_floors") or {})
    profile_floors = dict((cfg_floors.get("profiles") or {}).get(profile_name) or {})
    merged_floors = dict(floor_defaults.get(profile_name, floor_defaults["default"]))
    merged_floors.update({k: v for k, v in cfg_floors.items() if k != "profiles"})
    merged_floors.update(profile_floors)

    return {
        "name": profile_name,
        "source": source,
        "floors": merged_floors,
    }


def _resolve_reduction_floors(profile: dict[str, Any], n_species: int, n_reactions: int) -> dict[str, Any]:
    cfg_floors = dict(profile.get("floors") or {})
    min_species_abs = int(max(0, cfg_floors.get("min_species_abs", 0)))
    min_reactions_abs = int(max(0, cfg_floors.get("min_reactions_abs", 0)))
    min_species_ratio = float(max(0.0, cfg_floors.get("min_species_ratio", 0.0)))
    min_reactions_ratio = float(max(0.0, cfg_floors.get("min_reactions_ratio", 0.0)))

    min_species_after = max(min_species_abs, int(np.ceil(float(n_species) * min_species_ratio)))
    min_reactions_after = max(min_reactions_abs, int(np.ceil(float(n_reactions) * min_reactions_ratio)))
    min_species_after = int(min(max(min_species_after, 0), max(int(n_species), 0)))
    min_reactions_after = int(min(max(min_reactions_after, 0), max(int(n_reactions), 0)))
    return {
        "profile": str(profile.get("name", "default")),
        "source": str(profile.get("source", "unknown")),
        "min_species_abs": min_species_abs,
        "min_species_ratio": min_species_ratio,
        "min_reactions_abs": min_reactions_abs,
        "min_reactions_ratio": min_reactions_ratio,
        "min_species_after": min_species_after,
        "min_reactions_after": min_reactions_after,
    }


def _resolve_balance_bands(profile: dict[str, Any], species_before: int, reactions_before: int, cfg: dict[str, Any]) -> dict[str, Any]:
    default_by_profile = {
        "default": {
            "enabled": False,
            "balance_mode": "binary",
            "min_reaction_species_ratio": 0.0,
            "max_reaction_species_ratio": 1.0e9,
            "min_active_species_coverage": 0.0,
            "min_weighted_active_species_coverage": 0.0,
            "min_active_species_coverage_top_weighted": 0.0,
            "min_essential_species_coverage": 0.0,
            "max_cluster_size_ratio": 1.0,
            "top_weight_mass_ratio": 0.80,
            "min_nu_rank_ratio": 0.0,
        },
        "benchmarks_diamond": {
            "enabled": True,
            "balance_mode": "hybrid",
            "min_reaction_species_ratio": 0.40,
            "max_reaction_species_ratio": 4.00,
            "min_active_species_coverage": 0.50,
            "min_weighted_active_species_coverage": 0.80,
            "min_active_species_coverage_top_weighted": 0.80,
            "min_essential_species_coverage": 0.85,
            "max_cluster_size_ratio": 1.0,
            "top_weight_mass_ratio": 0.80,
            "min_nu_rank_ratio": 0.35,
        },
        "benchmark_sif4_sin3n4_cvd": {
            "enabled": True,
            "balance_mode": "hybrid",
            "min_reaction_species_ratio": 0.30,
            "max_reaction_species_ratio": 6.00,
            "min_active_species_coverage": 0.35,
            "min_weighted_active_species_coverage": 0.78,
            "min_active_species_coverage_top_weighted": 0.78,
            "min_essential_species_coverage": 0.80,
            "max_cluster_size_ratio": 1.0,
            "top_weight_mass_ratio": 0.80,
            "min_nu_rank_ratio": 0.30,
        },
        "benchmark_large": {
            "enabled": True,
            "balance_mode": "hybrid",
            "min_reaction_species_ratio": 0.30,
            "max_reaction_species_ratio": 6.00,
            "min_active_species_coverage": 0.40,
            "min_weighted_active_species_coverage": 0.80,
            "min_active_species_coverage_top_weighted": 0.80,
            "min_essential_species_coverage": 0.80,
            "max_cluster_size_ratio": 1.0,
            "top_weight_mass_ratio": 0.80,
            "min_nu_rank_ratio": 0.30,
        },
    }
    bal_cfg = dict(cfg.get("balance_constraints") or {})
    profile_name = str(profile.get("name", "default"))
    merged = dict(default_by_profile.get(profile_name, default_by_profile["default"]))
    merged.update({k: v for k, v in bal_cfg.items() if k != "profiles"})
    profiles_cfg = dict(bal_cfg.get("profiles") or {})
    merged.update(dict(profiles_cfg.get(profile_name) or {}))
    merged["enabled"] = bool(merged.get("enabled", False))
    merged["balance_mode"] = str(merged.get("balance_mode", "binary")).strip().lower() or "binary"
    merged["species_before"] = int(species_before)
    merged["reactions_before"] = int(reactions_before)
    merged["profile"] = profile_name
    return merged


def _resolve_dynamic_balance_bands(
    profile_name: str,
    base_bands: dict[str, Any],
    species_before: int,
    reactions_before: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    out = dict(base_bands)
    balance_cfg = dict(cfg.get("balance_constraints") or {})
    dynamic_cfg = dict(balance_cfg.get("dynamic") or {})
    profile_dynamic = dict((dynamic_cfg.get("profiles") or {}).get(profile_name) or {})
    merged_dynamic = {k: v for k, v in dynamic_cfg.items() if k != "profiles"}
    merged_dynamic.update(profile_dynamic)

    enabled = bool(merged_dynamic.get("enabled", False))
    complexity_offset = float(merged_dynamic.get("complexity_offset_reactions", 100.0))
    complexity_span = float(merged_dynamic.get("complexity_span_reactions", 300.0))
    complexity_span = max(abs(complexity_span), 1.0e-12)
    complexity_min = float(merged_dynamic.get("complexity_min", 0.0))
    complexity_max = float(merged_dynamic.get("complexity_max", 1.0))
    complexity = float(
        np.clip(
            (float(reactions_before) - complexity_offset) / complexity_span,
            complexity_min,
            complexity_max,
        )
    )

    default_dynamic_rules = {
        "benchmarks_diamond": {
            "min_reaction_species_ratio": {"slope": -0.08, "floor": 0.30},
            "max_reaction_species_ratio": {"slope": 2.00, "ceiling": 6.00},
            "min_active_species_coverage": {"slope": -0.14, "floor": 0.38},
        },
        "benchmark_sif4_sin3n4_cvd": {
            "min_reaction_species_ratio": {"slope": -0.04, "floor": 0.24},
            "max_reaction_species_ratio": {"slope": 1.00, "ceiling": 7.00},
            "min_active_species_coverage": {"slope": -0.06, "floor": 0.30},
        },
        "benchmark_large": {
            "min_reaction_species_ratio": {"slope": -0.04, "floor": 0.24},
            "max_reaction_species_ratio": {"slope": 1.50, "ceiling": 8.00},
            "min_active_species_coverage": {"slope": -0.08, "floor": 0.32},
        },
    }
    rule_overrides = dict(merged_dynamic.get("rules") or {})
    if not rule_overrides:
        # Backward-compatible direct per-key overrides under dynamic.
        for key in (
            "min_reaction_species_ratio",
            "max_reaction_species_ratio",
            "min_active_species_coverage",
        ):
            raw = merged_dynamic.get(key)
            if isinstance(raw, dict):
                rule_overrides[key] = dict(raw)

    rules = dict(default_dynamic_rules.get(profile_name, {}))
    for key, val in rule_overrides.items():
        if isinstance(val, dict):
            merged_rule = dict(rules.get(key) or {})
            merged_rule.update(val)
            rules[key] = merged_rule

    def _apply_rule(base: float, key: str, floor_default: float, ceil_default: float) -> float:
        rule = dict(rules.get(key) or {})
        slope = float(rule.get("slope", 0.0))
        value = float(base + slope * complexity)
        floor_v = float(rule.get("floor", floor_default))
        ceil_v = float(rule.get("ceiling", rule.get("cap", ceil_default)))
        if np.isfinite(floor_v):
            value = max(value, floor_v)
        if np.isfinite(ceil_v):
            value = min(value, ceil_v)
        return float(value)

    if enabled:
        out["min_reaction_species_ratio"] = _apply_rule(
            float(out.get("min_reaction_species_ratio", 0.0)),
            "min_reaction_species_ratio",
            0.0,
            1.0e9,
        )
        out["max_reaction_species_ratio"] = _apply_rule(
            float(out.get("max_reaction_species_ratio", 1.0e9)),
            "max_reaction_species_ratio",
            0.0,
            1.0e9,
        )
        out["min_active_species_coverage"] = _apply_rule(
            float(out.get("min_active_species_coverage", 0.0)),
            "min_active_species_coverage",
            0.0,
            1.0,
        )

    out["balance_dynamic_applied"] = bool(enabled)
    out["balance_dynamic_complexity"] = float(complexity if enabled else 0.0)
    out["balance_dynamic_profile"] = str(profile_name)
    out["rs_upper_effective"] = float(out.get("max_reaction_species_ratio", 1.0e9))
    out["active_cov_effective_floor"] = float(out.get("min_active_species_coverage", 0.0))
    return out


def _compute_balance_margin_vector(metrics: dict[str, float], bands: dict[str, Any]) -> dict[str, Any]:
    ratio = float(metrics.get("reaction_species_ratio", 0.0))
    coverage = float(metrics.get("active_species_coverage", 0.0))
    weighted_cov = float(metrics.get("weighted_active_species_coverage", coverage))
    top_weighted_cov = float(metrics.get("active_species_coverage_top_weighted", weighted_cov))
    essential_cov = float(metrics.get("essential_species_coverage", 1.0))
    nu_rank_ratio = float(metrics.get("nu_rank_ratio", 0.0))
    max_cluster_ratio = float(metrics.get("max_cluster_size_ratio", 0.0))

    min_ratio = float(bands.get("min_reaction_species_ratio", 0.0))
    max_ratio = float(bands.get("max_reaction_species_ratio", 1.0e9))
    min_cov = float(bands.get("min_active_species_coverage", 0.0))
    min_weighted_cov = float(bands.get("min_weighted_active_species_coverage", 0.0))
    min_top_weighted_cov = float(bands.get("min_active_species_coverage_top_weighted", 0.0))
    min_essential_cov = float(bands.get("min_essential_species_coverage_effective", bands.get("min_essential_species_coverage", 0.0)))
    min_rank = float(bands.get("min_nu_rank_ratio", 0.0))
    max_cluster_size_ratio = float(bands.get("max_cluster_size_ratio", 1.0e9))

    ratio_lower_margin = ratio - min_ratio
    ratio_upper_margin = max_ratio - ratio
    coverage_margin_terms: list[float] = []
    margin_terms = [ratio_lower_margin, ratio_upper_margin, nu_rank_ratio - min_rank]
    balance_mode = str(bands.get("balance_mode", "binary"))

    coverage_margin = coverage - min_cov
    if balance_mode in {"binary", "hybrid"}:
        margin_terms.append(coverage_margin)
        coverage_margin_terms.append(coverage_margin)
    weighted_margin = weighted_cov - min_weighted_cov
    top_weighted_margin = top_weighted_cov - min_top_weighted_cov
    if balance_mode in {"weighted", "hybrid"}:
        margin_terms.extend([weighted_margin, top_weighted_margin])
        coverage_margin_terms.extend([weighted_margin, top_weighted_margin])
    essential_margin = essential_cov - min_essential_cov
    if balance_mode in {"essential", "hybrid"} and min_essential_cov > 0.0:
        margin_terms.append(essential_margin)
    cluster_size_margin = max_cluster_size_ratio - max_cluster_ratio
    if max_cluster_size_ratio < 1.0e8:
        margin_terms.append(cluster_size_margin)

    return {
        "balance_margin": float(min(margin_terms)) if margin_terms else 0.0,
        "coverage_margin": float(min(coverage_margin_terms)) if coverage_margin_terms else 0.0,
        "cluster_size_margin": float(cluster_size_margin),
        "detail": {
            "reaction_species_ratio": {
                "value": ratio,
                "min_required": min_ratio,
                "max_allowed": max_ratio,
                "min_margin": ratio_lower_margin,
                "max_margin": ratio_upper_margin,
            },
            "active_species_coverage": {
                "value": coverage,
                "min_required": min_cov,
                "margin": coverage_margin,
            },
            "weighted_active_species_coverage": {
                "value": weighted_cov,
                "min_required": min_weighted_cov,
                "margin": weighted_margin,
            },
            "active_species_coverage_top_weighted": {
                "value": top_weighted_cov,
                "min_required": min_top_weighted_cov,
                "margin": top_weighted_margin,
            },
            "essential_species_coverage": {
                "value": essential_cov,
                "min_required": min_essential_cov,
                "margin": essential_margin,
            },
            "nu_rank_ratio": {
                "value": nu_rank_ratio,
                "min_required": min_rank,
                "margin": nu_rank_ratio - min_rank,
            },
            "max_cluster_size_ratio": {
                "value": max_cluster_ratio,
                "max_allowed": max_cluster_size_ratio,
                "margin": cluster_size_margin,
            },
            "dynamic": {
                "applied": bool(bands.get("balance_dynamic_applied", False)),
                "complexity": float(bands.get("balance_dynamic_complexity", 0.0)),
                "profile": str(bands.get("balance_dynamic_profile", "")),
            },
        },
    }


def _build_species_activity_weights(
    X: np.ndarray | None,
    wdot: np.ndarray | None,
    F_bar: np.ndarray | None,
    cfg: dict[str, Any],
) -> np.ndarray:
    def _norm(v: np.ndarray) -> np.ndarray:
        arr = np.maximum(np.asarray(v, dtype=float), 0.0)
        total = float(np.sum(arr))
        if total <= 0.0:
            return np.zeros_like(arr)
        return arr / total

    bal_cfg = dict(cfg.get("balance_constraints") or {})
    if not bal_cfg and ("activity_weights" in cfg):
        bal_cfg = dict(cfg)
    weight_cfg = dict(bal_cfg.get("activity_weights") or {})
    wx = float(weight_cfg.get("x", 0.25))
    ww = float(weight_cfg.get("wdot", 0.55))
    wf = float(weight_cfg.get("flux", 0.20))
    norm = max(1.0e-12, wx + ww + wf)
    wx /= norm
    ww /= norm
    wf /= norm

    n_species = 0
    x_arr = None if X is None else np.asarray(X, dtype=float)
    wdot_arr = None if wdot is None else np.asarray(wdot, dtype=float)
    f_arr = None if F_bar is None else np.asarray(F_bar, dtype=float)
    if x_arr is not None and x_arr.ndim == 2:
        n_species = max(n_species, int(x_arr.shape[1]))
    if wdot_arr is not None and wdot_arr.ndim == 2:
        n_species = max(n_species, int(wdot_arr.shape[1]))
    if f_arr is not None and f_arr.ndim == 2:
        n_species = max(n_species, int(f_arr.shape[0]))
    if n_species <= 0:
        return np.zeros((0,), dtype=float)

    x_term = np.zeros((n_species,), dtype=float)
    if x_arr is not None and x_arr.ndim == 2 and x_arr.shape[1] > 0:
        cols = min(n_species, int(x_arr.shape[1]))
        x_term[:cols] = np.mean(np.abs(x_arr[:, :cols]), axis=0)
    wdot_term = np.zeros((n_species,), dtype=float)
    if wdot_arr is not None and wdot_arr.ndim == 2 and wdot_arr.shape[1] > 0:
        cols = min(n_species, int(wdot_arr.shape[1]))
        wdot_term[:cols] = np.mean(np.abs(wdot_arr[:, :cols]), axis=0)
    flux_term = np.zeros((n_species,), dtype=float)
    if f_arr is not None and f_arr.ndim == 2 and f_arr.shape[0] > 0 and f_arr.shape[1] > 0:
        rows = min(n_species, int(f_arr.shape[0]))
        cols = min(n_species, int(f_arr.shape[1]))
        flux_term[:rows] = np.mean(np.abs(f_arr[:rows, :cols]), axis=1)

    combined = (wx * _norm(x_term)) + (ww * _norm(wdot_term)) + (wf * _norm(flux_term))
    total = float(np.sum(combined))
    if total <= 0.0:
        return np.full((n_species,), 1.0 / float(n_species), dtype=float)
    return np.asarray(combined / total, dtype=float)


def _resolve_essential_species(qoi_cfg: dict[str, Any], species_meta: list[dict[str, Any]], cfg: dict[str, Any]) -> set[str]:
    bal_cfg = dict(cfg.get("balance_constraints") or {})
    names: set[str] = set()
    if "essential_qoi_keys" in bal_cfg:
        qoi_keys = list(bal_cfg.get("essential_qoi_keys") or [])
    else:
        qoi_keys = ["species_last", "deposition_integral"]
    for key in qoi_keys:
        values = list(qoi_cfg.get(key) or [])
        for item in values:
            token = str(item).strip()
            if token:
                names.add(token)

    if bool(bal_cfg.get("include_selector_species_in_essential", True)):
        selectors = list(qoi_cfg.get("selectors") or [])
        for sel in selectors:
            token = str(sel).strip()
            if not token or ":" not in token:
                continue
            parts = token.split(":")
            if len(parts) >= 2:
                sp = parts[1].strip()
                if sp:
                    names.add(sp)

    for item in list(bal_cfg.get("essential_species") or []):
        token = str(item).strip()
        if token:
            names.add(token)

    role_set = {str(x).strip().lower() for x in list(bal_cfg.get("essential_roles") or []) if str(x).strip()}
    if role_set:
        for meta in species_meta:
            role = str(meta.get("role", "")).strip().lower()
            if role and role in role_set:
                token = str(meta.get("name", "")).strip()
                if token:
                    names.add(token)

    pattern_tokens = [str(x).strip() for x in list(bal_cfg.get("essential_name_patterns") or []) if str(x).strip()]
    if pattern_tokens:
        names_l = {n.lower() for n in names}
        for meta in species_meta:
            name = str(meta.get("name", "")).strip()
            if not name:
                continue
            lname = name.lower()
            if lname in names_l:
                continue
            if any(p.lower() in lname for p in pattern_tokens):
                names.add(name)
                names_l.add(lname)

    available = {str(m.get("name", "")).strip() for m in species_meta}
    lower_to_name = {name.lower(): name for name in available if name}
    resolved: set[str] = set()
    for token in names:
        if token in available:
            resolved.add(token)
            continue
        mapped = lower_to_name.get(token.lower())
        if mapped:
            resolved.add(mapped)

    max_essential = int(max(0, bal_cfg.get("max_essential_species", 0)))
    if max_essential > 0 and len(resolved) > max_essential:
        priority: list[str] = []
        for key in qoi_keys:
            for item in list(qoi_cfg.get(key) or []):
                token = str(item).strip()
                if token:
                    priority.append(token)
        priority.extend([str(x).strip() for x in list(bal_cfg.get("essential_species") or []) if str(x).strip()])
        picked: set[str] = set()
        for token in priority:
            cand = token if token in available else lower_to_name.get(token.lower())
            if cand and cand in resolved:
                picked.add(cand)
            if len(picked) >= max_essential:
                break
        if len(picked) < max_essential:
            for token in sorted(resolved):
                picked.add(token)
                if len(picked) >= max_essential:
                    break
        resolved = picked
    return resolved


def _build_essential_cluster_mask(S: np.ndarray, species_meta: list[dict[str, Any]], essential_species: set[str]) -> np.ndarray:
    s_arr = np.asarray(S, dtype=float)
    if s_arr.ndim != 2 or s_arr.shape[1] <= 0:
        return np.zeros((0,), dtype=bool)
    mask = np.zeros((s_arr.shape[1],), dtype=bool)
    if not essential_species:
        return mask

    names = [str(m.get("name", "")).strip() for m in species_meta]
    for i, name in enumerate(names):
        if name not in essential_species:
            continue
        if i >= s_arr.shape[0]:
            continue
        row = np.asarray(s_arr[i, :], dtype=float)
        active = np.where(row > 0.5)[0]
        if active.size == 0:
            idx = int(np.argmax(row))
            mask[idx] = True
        else:
            mask[active] = True
    return mask


def _compute_structural_balance_metrics(
    *,
    nu_reduced: np.ndarray,
    species_after: int,
    reactions_after: int,
    activity_weights: np.ndarray | None = None,
    essential_cluster_mask: np.ndarray | None = None,
    S_reduced: np.ndarray | None = None,
    species_before: int | None = None,
    top_weight_mass_ratio: float = 0.80,
    balance_mode: str = "binary",
) -> dict[str, float]:
    ns = int(max(species_after, 0))
    nr = int(max(reactions_after, 0))
    if ns <= 0:
        return {
            "reaction_species_ratio": 0.0,
            "active_species_coverage": 0.0,
            "weighted_active_species_coverage": 0.0,
            "active_species_coverage_top_weighted": 0.0,
            "essential_species_coverage": 1.0,
            "nu_rank_ratio": 0.0,
            "max_cluster_size_ratio": 0.0,
            "active_species_count": 0.0,
            "balance_mode": str(balance_mode),
        }
    ratio = float(nr) / float(max(ns, 1))

    arr = np.asarray(nu_reduced, dtype=float)
    if arr.ndim != 2:
        arr = np.zeros((ns, nr), dtype=float)
    if arr.shape[0] != ns:
        arr2 = np.zeros((ns, arr.shape[1] if arr.ndim == 2 else nr), dtype=float)
        copy_rows = min(arr2.shape[0], arr.shape[0] if arr.ndim == 2 else 0)
        if copy_rows > 0 and arr.ndim == 2:
            arr2[:copy_rows, : arr.shape[1]] = arr[:copy_rows, :]
        arr = arr2
    if arr.shape[1] != nr:
        arr2 = np.zeros((arr.shape[0], nr), dtype=float)
        copy_cols = min(arr.shape[1], nr)
        if copy_cols > 0:
            arr2[:, :copy_cols] = arr[:, :copy_cols]
        arr = arr2

    if nr <= 0:
        active_mask = np.zeros((ns,), dtype=bool)
        rank_ratio = 0.0
    else:
        active_mask = np.any(np.abs(arr) > 1.0e-12, axis=1)
        rank = float(np.linalg.matrix_rank(arr, tol=1.0e-10))
        rank_ratio = float(rank / float(max(1, min(arr.shape[0], arr.shape[1]))))
    active_species = int(np.sum(active_mask))
    coverage = float(active_species) / float(max(ns, 1))

    weights = np.asarray(activity_weights, dtype=float) if activity_weights is not None else np.zeros((0,), dtype=float)
    if weights.shape != (ns,):
        weights_fixed = np.zeros((ns,), dtype=float)
        copy_n = min(ns, int(weights.shape[0])) if weights.ndim == 1 else 0
        if copy_n > 0 and weights.ndim == 1:
            weights_fixed[:copy_n] = np.maximum(weights[:copy_n], 0.0)
        weights = weights_fixed
    weights_sum = float(np.sum(weights))
    if weights_sum > 0.0:
        weighted_coverage = float(np.sum(weights[active_mask])) / weights_sum
    else:
        weighted_coverage = coverage
    top_mass_ratio = float(np.clip(top_weight_mass_ratio, 0.0, 1.0))
    if weights_sum > 0.0 and top_mass_ratio > 0.0:
        order = np.argsort(-weights)
        accum = 0.0
        target_mass = top_mass_ratio * weights_sum
        top_mask = np.zeros((ns,), dtype=bool)
        for idx in order:
            idx_i = int(idx)
            if weights[idx_i] <= 0.0:
                break
            top_mask[idx_i] = True
            accum += float(weights[idx_i])
            if accum >= target_mass:
                break
        top_weights_sum = float(np.sum(weights[top_mask]))
        if top_weights_sum > 0.0:
            top_weighted_cov = float(np.sum(weights[np.logical_and(top_mask, active_mask)])) / top_weights_sum
        else:
            top_weighted_cov = weighted_coverage
    else:
        top_weighted_cov = weighted_coverage

    essential_mask = (
        np.asarray(essential_cluster_mask, dtype=bool) if essential_cluster_mask is not None else np.zeros((ns,), dtype=bool)
    )
    if essential_mask.shape != (ns,):
        fixed = np.zeros((ns,), dtype=bool)
        copy_n = min(ns, int(essential_mask.shape[0])) if essential_mask.ndim == 1 else 0
        if copy_n > 0 and essential_mask.ndim == 1:
            fixed[:copy_n] = essential_mask[:copy_n]
        essential_mask = fixed
    essential_count = int(np.sum(essential_mask))
    if essential_count <= 0:
        essential_cov = 1.0
    else:
        essential_active = np.logical_and(active_mask, essential_mask)
        essential_weights = weights[essential_mask]
        if essential_weights.size > 0 and float(np.sum(essential_weights)) > 0.0:
            essential_cov = float(np.sum(weights[essential_active])) / float(np.sum(essential_weights))
        else:
            essential_cov = float(np.sum(essential_active)) / float(max(essential_count, 1))

    s_arr = np.asarray(S_reduced, dtype=float) if S_reduced is not None else np.zeros((0, 0), dtype=float)
    if s_arr.ndim == 2 and s_arr.shape[1] > 0:
        sizes = np.sum(s_arr > 0.5, axis=0).astype(float)
        denom = int(species_before) if species_before is not None else int(s_arr.shape[0])
        max_cluster_ratio = float(np.max(sizes) / float(max(1, denom))) if sizes.size else 0.0
    else:
        max_cluster_ratio = 0.0

    return {
        "reaction_species_ratio": ratio,
        "active_species_coverage": coverage,
        "weighted_active_species_coverage": float(np.clip(weighted_coverage, 0.0, 1.0)),
        "active_species_coverage_top_weighted": float(np.clip(top_weighted_cov, 0.0, 1.0)),
        "essential_species_coverage": float(np.clip(essential_cov, 0.0, 1.0)),
        "nu_rank_ratio": rank_ratio,
        "max_cluster_size_ratio": float(max(0.0, max_cluster_ratio)),
        "active_species_count": float(active_species),
        "essential_species_count": float(essential_count),
        "balance_mode": str(balance_mode),
    }


def _evaluate_balance_gate(metrics: dict[str, float], bands: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(bands.get("enabled", False))
    balance_mode = str(bands.get("balance_mode", "binary")).strip().lower() or "binary"
    ratio = float(metrics.get("reaction_species_ratio", 0.0))
    coverage = float(metrics.get("active_species_coverage", 0.0))
    weighted_cov = float(metrics.get("weighted_active_species_coverage", coverage))
    top_weighted_cov = float(metrics.get("active_species_coverage_top_weighted", weighted_cov))
    essential_cov = float(metrics.get("essential_species_coverage", 1.0))
    nu_rank_ratio = float(metrics.get("nu_rank_ratio", 0.0))
    max_cluster_ratio = float(metrics.get("max_cluster_size_ratio", 0.0))
    min_ratio = float(bands.get("min_reaction_species_ratio", 0.0))
    max_ratio = float(bands.get("max_reaction_species_ratio", 1.0e9))
    min_cov = float(bands.get("min_active_species_coverage", 0.0))
    min_weighted_cov = float(bands.get("min_weighted_active_species_coverage", 0.0))
    min_top_weighted_cov = float(bands.get("min_active_species_coverage_top_weighted", 0.0))
    min_essential_cov = float(bands.get("min_essential_species_coverage", 0.0))
    max_cluster_size_ratio = float(bands.get("max_cluster_size_ratio", 1.0e9))
    relax_essential = bool(bands.get("essential_relax_when_weighted_passed", False))
    min_essential_cov_relaxed = float(bands.get("min_essential_species_coverage_relaxed", min_essential_cov))
    min_rank = float(bands.get("min_nu_rank_ratio", 0.0))

    effective_min_essential_cov = min_essential_cov
    if (
        balance_mode in {"essential", "hybrid"}
        and relax_essential
        and weighted_cov >= min_weighted_cov
        and coverage >= min_cov
    ):
        effective_min_essential_cov = min(min_essential_cov, min_essential_cov_relaxed)

    violations: list[str] = []
    if ratio < min_ratio:
        violations.append("min_reaction_species_ratio")
    if ratio > max_ratio:
        violations.append("max_reaction_species_ratio")
    if balance_mode in {"binary", "hybrid"} and coverage < min_cov:
        violations.append("min_active_species_coverage")
    if balance_mode in {"weighted", "hybrid"} and weighted_cov < min_weighted_cov:
        violations.append("min_weighted_active_species_coverage")
    if balance_mode in {"weighted", "hybrid"} and top_weighted_cov < min_top_weighted_cov:
        violations.append("min_active_species_coverage_top_weighted")
    if (
        balance_mode in {"essential", "hybrid"}
        and effective_min_essential_cov > 0.0
        and essential_cov < effective_min_essential_cov
    ):
        violations.append("min_essential_species_coverage")
    if max_cluster_ratio > max_cluster_size_ratio:
        violations.append("max_cluster_size_ratio")
    if nu_rank_ratio < min_rank:
        violations.append("min_nu_rank_ratio")

    if not enabled:
        violations = []
    return {
        "enabled": bool(enabled),
        "balance_mode": balance_mode,
        "passed": (len(violations) == 0),
        "violations": violations,
        "metrics": metrics,
        "bands": {
            "balance_mode": balance_mode,
            "min_reaction_species_ratio": min_ratio,
            "max_reaction_species_ratio": max_ratio,
            "min_active_species_coverage": min_cov,
            "min_weighted_active_species_coverage": min_weighted_cov,
            "min_active_species_coverage_top_weighted": min_top_weighted_cov,
            "min_essential_species_coverage": min_essential_cov,
            "min_essential_species_coverage_effective": effective_min_essential_cov,
            "essential_relax_when_weighted_passed": relax_essential,
            "min_essential_species_coverage_relaxed": min_essential_cov_relaxed,
            "max_cluster_size_ratio": max_cluster_size_ratio,
            "min_nu_rank_ratio": min_rank,
            "balance_dynamic_applied": bool(bands.get("balance_dynamic_applied", False)),
            "balance_dynamic_complexity": float(bands.get("balance_dynamic_complexity", 0.0)),
            "balance_dynamic_profile": str(bands.get("balance_dynamic_profile", "")),
            "rs_upper_effective": float(bands.get("rs_upper_effective", max_ratio)),
            "active_cov_effective_floor": float(bands.get("active_cov_effective_floor", min_cov)),
        },
    }


def _stage_selection_score(row: dict[str, Any], floors: dict[str, Any], weights: dict[str, float]) -> float:
    species_before = max(int(row.get("species_before", 1)), 1)
    reactions_before = max(int(row.get("reactions_before", 1)), 1)
    species_after = max(int(row.get("species_after", 0)), 0)
    reactions_after = max(int(row.get("reactions_after", 0)), 0)
    mean_rel_diff = float(row.get("mean_rel_diff", 1.0))

    max_mean_rel = float(row.get("_selection_max_mean_rel", 0.40))
    err_term = 1.0 - float(np.clip(mean_rel_diff / max(max_mean_rel, 1.0e-12), 0.0, 2.0))
    reaction_comp = 1.0 - float(reactions_after) / float(reactions_before)
    species_comp = 1.0 - float(species_after) / float(species_before)

    margin_species = float(species_after - int(floors.get("min_species_after", 0))) / float(species_before)
    margin_reactions = float(reactions_after - int(floors.get("min_reactions_after", 0))) / float(reactions_before)
    floor_margin = float(max(0.0, min(margin_species, margin_reactions)))
    balance_margin = float(max(0.0, row.get("balance_margin", 0.0)))
    coverage_margin = float(max(0.0, row.get("coverage_margin", balance_margin)))
    cluster_size_margin = float(max(0.0, row.get("cluster_size_margin", 0.0)))

    score = (
        float(weights.get("err", 0.55)) * err_term
        + float(weights.get("reaction_comp", 0.30)) * reaction_comp
        + float(weights.get("species_comp", 0.10)) * species_comp
        + float(weights.get("floor_margin", 0.05)) * floor_margin
        + float(weights.get("balance_margin", 0.05)) * balance_margin
        + float(weights.get("coverage_margin", 0.05)) * coverage_margin
        + float(weights.get("cluster_size_margin", 0.03)) * cluster_size_margin
    )
    if not bool(row.get("floor_passed", True)):
        score -= 5.0
    if not bool(row.get("balance_gate_passed", True)):
        score -= 3.0
    if not bool(row.get("gate_passed", False)):
        score -= 2.0
    if bool(row.get("physical_degraded", False)):
        score -= 0.25
    return float(score)


def _pareto_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    def obj(r: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(r.get("species_after", float("inf"))),
            float(r.get("reactions_after", float("inf"))),
            float(r.get("mean_rel_diff", float("inf"))),
        )

    objs = [obj(r) for r in rows]
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            o = objs[j]
            t = objs[i]
            if (o[0] <= t[0] and o[1] <= t[1] and o[2] <= t[2]) and (o != t):
                dominated = True
                break
        if not dominated:
            out.append(row)
    return out


def _selection_weights_from_cfg(selection_cfg: dict[str, Any]) -> dict[str, float]:
    weights_raw = dict(selection_cfg.get("weights") or {})
    return {
        "err": float(weights_raw.get("err", 0.55)),
        "reaction_comp": float(weights_raw.get("reaction_comp", 0.30)),
        "species_comp": float(weights_raw.get("species_comp", 0.10)),
        "floor_margin": float(weights_raw.get("floor_margin", 0.05)),
        "balance_margin": float(weights_raw.get("balance_margin", 0.05)),
        "coverage_margin": float(weights_raw.get("coverage_margin", 0.05)),
        "cluster_size_margin": float(weights_raw.get("cluster_size_margin", 0.03)),
    }


def _tie_breaker_sort_key(row: dict[str, Any], tie_breakers: list[str]) -> tuple[float, ...]:
    out: list[float] = []
    reactions_before = max(int(row.get("reactions_before", 1)), 1)
    reactions_after = max(int(row.get("reactions_after", 0)), 0)
    species_before = max(int(row.get("species_before", 1)), 1)
    species_after = max(int(row.get("species_after", 0)), 0)
    mean_rel_diff = float(row.get("mean_rel_diff", 1.0))
    for name in tie_breakers:
        key = str(name).strip().lower()
        if key == "reaction_reduction":
            out.append(-(1.0 - float(reactions_after) / float(reactions_before)))
        elif key == "species_reduction":
            out.append(-(1.0 - float(species_after) / float(species_before)))
        elif key == "mean_rel_diff":
            out.append(mean_rel_diff)
        else:
            out.append(float(row.get(key, 0.0)))
    return tuple(out)


def _select_stage_pass_first(stage_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    selection_cfg = dict(cfg.get("selection") or {})
    weights = _selection_weights_from_cfg(selection_cfg)
    tie_breakers = [
        str(x).strip()
        for x in list(selection_cfg.get("tie_breakers") or ["reaction_reduction", "species_reduction", "mean_rel_diff"])
        if str(x).strip()
    ]

    for row in stage_rows:
        floors = dict(row.get("_floors") or {})
        row["selection_score"] = _stage_selection_score(row, floors, weights)

    passed_non_degraded = [
        r
        for r in stage_rows
        if bool(r.get("gate_passed"))
        and bool(r.get("floor_passed", True))
        and bool(r.get("balance_gate_passed", True))
        and bool(r.get("cluster_guard_passed", True))
        and not bool(r.get("physical_degraded", False))
    ]
    passed = [
        r
        for r in stage_rows
        if bool(r.get("gate_passed"))
        and bool(r.get("floor_passed", True))
        and bool(r.get("balance_gate_passed", True))
        and bool(r.get("cluster_guard_passed", True))
    ]
    floor_non_degraded = [
        r
        for r in stage_rows
        if bool(r.get("floor_passed", True))
        and bool(r.get("balance_gate_passed", True))
        and not bool(r.get("physical_degraded", False))
    ]
    floor_rows = [r for r in stage_rows if bool(r.get("floor_passed", True)) and bool(r.get("balance_gate_passed", True))]
    non_degraded = [r for r in stage_rows if not bool(r.get("physical_degraded", False))]

    candidate_pool = passed_non_degraded or passed or floor_non_degraded or floor_rows or non_degraded or stage_rows
    pareto = _pareto_rows(candidate_pool)
    if not pareto:
        pareto = list(candidate_pool)
    pareto.sort(
        key=lambda r: (
            -float(r.get("selection_score", -1.0e9)),
            *_tie_breaker_sort_key(r, tie_breakers),
            int(r.get("reactions_after", 1.0e9)),
            int(r.get("species_after", 1.0e9)),
        )
    )
    selected = pareto[0]
    return {
        "selected": selected,
        "pareto_candidates": pareto,
        "weights": weights,
        "policy": "pass_first_pareto",
        "tie_breakers": tie_breakers,
    }


def _select_stage_physics_first(stage_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    if not stage_rows:
        raise ValueError("stage_rows is empty")

    selection_cfg = dict(cfg.get("selection") or {})
    policy = str(selection_cfg.get("policy", "physics_first_pareto"))
    weights = _selection_weights_from_cfg(selection_cfg)

    if policy == "pass_first_pareto":
        return _select_stage_pass_first(stage_rows, cfg)

    if policy != "physics_first_pareto":
        selected = _choose_stage(stage_rows)
        selected["selection_score"] = _stage_selection_score(selected, dict(selected.get("_floors") or {}), weights)
        return {"selected": selected, "pareto_candidates": [selected], "weights": weights, "policy": policy}

    passed_floor_non_degraded = [
        r
        for r in stage_rows
        if bool(r.get("gate_passed"))
        and bool(r.get("floor_passed", True))
        and bool(r.get("balance_gate_passed", True))
        and not bool(r.get("physical_degraded", False))
    ]
    passed_floor = [
        r
        for r in stage_rows
        if bool(r.get("gate_passed")) and bool(r.get("floor_passed", True)) and bool(r.get("balance_gate_passed", True))
    ]
    floor_non_degraded = [
        r
        for r in stage_rows
        if bool(r.get("floor_passed", True))
        and bool(r.get("balance_gate_passed", True))
        and not bool(r.get("physical_degraded", False))
    ]
    floor_rows = [r for r in stage_rows if bool(r.get("floor_passed", True)) and bool(r.get("balance_gate_passed", True))]
    non_degraded = [r for r in stage_rows if not bool(r.get("physical_degraded", False))]

    candidate_pool = passed_floor_non_degraded or passed_floor or floor_non_degraded or floor_rows or non_degraded or stage_rows
    pareto = _pareto_rows(candidate_pool)
    if not pareto:
        pareto = list(candidate_pool)

    for row in stage_rows:
        floors = dict(row.get("_floors") or {})
        row["selection_score"] = _stage_selection_score(row, floors, weights)
    pareto.sort(
        key=lambda r: (
            -float(r.get("selection_score", -1.0e9)),
            float(r.get("mean_rel_diff", 1.0e9)),
            int(r.get("reactions_after", 1.0e9)),
            int(r.get("species_after", 1.0e9)),
        )
    )
    selected = pareto[0]

    return {"selected": selected, "pareto_candidates": pareto, "weights": weights, "policy": policy}


def _resolve_kfold_count(n_cases: int, cfg: dict[str, Any]) -> int:
    default_k = int(cfg.get("default_k", 2))
    default_k = max(2, default_k)
    k_map_raw = dict(cfg.get("k_by_case_count") or {})
    selected = default_k
    candidates: list[tuple[int, int]] = []
    for raw_k, raw_v in k_map_raw.items():
        try:
            case_key = int(raw_k)
            k_val = int(raw_v)
        except (TypeError, ValueError):
            continue
        if case_key <= 0:
            continue
        candidates.append((case_key, max(2, k_val)))
    candidates.sort(key=lambda x: x[0])
    for case_key, k_val in candidates:
        if n_cases >= case_key:
            selected = k_val
    return int(max(2, min(selected, max(n_cases, 2))))


def _resolve_surrogate_split_adaptive(conditions: list[dict[str, Any]], eval_cfg: dict[str, Any]) -> dict[str, Any]:
    split_cfg = dict(eval_cfg.get("surrogate_split") or {})
    policy = dict(split_cfg.get("kfold_policy") or {})
    case_ids = [str(c.get("case_id")) for c in conditions]
    n_cases = len(case_ids)
    min_cases_for_kfold = int(policy.get("min_cases_for_kfold", 4))
    mode = "kfold"
    fallback_reason: str | None = None
    if n_cases < max(2, min_cases_for_kfold):
        mode = "in_sample"
        fallback_reason = "insufficient_cases_for_kfold"
        return {
            "requested_mode": "adaptive_kfold",
            "mode": mode,
            "fallback_reason": fallback_reason,
            "case_count": int(n_cases),
            "effective_kfolds": 0,
            "fold_sizes": [],
            "folds": [],
        }

    split_cfg["kfolds"] = _resolve_kfold_count(n_cases, policy)
    return {
        "requested_mode": "adaptive_kfold",
        "mode": "kfold",
        "fallback_reason": None,
        "case_count": int(n_cases),
        "effective_kfolds": int(split_cfg["kfolds"]),
        "fold_sizes": [],
        "folds": [],
    }


def _resolve_surrogate_split(conditions: list[dict[str, Any]], eval_cfg: dict[str, Any]) -> dict[str, Any]:
    split_cfg = dict(eval_cfg.get("surrogate_split") or {})
    requested = str(split_cfg.get("mode", "kfold"))
    if requested not in {"kfold", "in_sample", "adaptive_kfold"}:
        raise ValueError("evaluation.surrogate_split.mode must be 'kfold', 'in_sample', or 'adaptive_kfold'")

    case_ids = [str(c.get("case_id")) for c in conditions]
    n_cases = len(case_ids)
    mode = requested
    fallback_reason: str | None = None

    if requested == "adaptive_kfold":
        adaptive = _resolve_surrogate_split_adaptive(conditions, eval_cfg)
        mode = str(adaptive.get("mode", "in_sample"))
        fallback_reason = adaptive.get("fallback_reason")
        split_cfg["kfolds"] = int(adaptive.get("effective_kfolds", split_cfg.get("kfolds", 2)))
    elif requested == "kfold" and n_cases < 4:
        mode = "in_sample"
        fallback_reason = "insufficient_cases_for_kfold"

    folds: list[list[str]] = []
    effective_kfolds = 0
    if mode == "kfold":
        k = int(split_cfg.get("kfolds", 2))
        k = max(2, min(k, n_cases))
        effective_kfolds = int(k)
        ordered = sorted(case_ids)
        for fold_idx in range(k):
            test_ids = [cid for i, cid in enumerate(ordered) if i % k == fold_idx]
            if test_ids:
                folds.append(test_ids)
        if len(folds) < 2:
            mode = "in_sample"
            fallback_reason = "degenerate_kfold_split"
            effective_kfolds = 0

    return {
        "requested_mode": requested,
        "mode": mode,
        "fallback_reason": fallback_reason,
        "case_count": int(n_cases),
        "effective_kfolds": int(effective_kfolds if mode == "kfold" else 0),
        "fold_sizes": [int(len(fold)) for fold in folds],
        "folds": folds,
    }


def _evaluate_surrogate_stage(
    *,
    surrogate_model_name: str,
    conditions: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    qoi_cfg: dict[str, Any],
    eval_cfg: dict[str, Any],
    split_plan: dict[str, Any],
    surrogate_l2: float,
    surrogate_blend: float,
    metric_drift: float,
    surrogate_gain: float,
    stage_idx: int,
    seed: int,
    trained_surrogate: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], Any, list[dict[str, Any]]]:
    perturb_scale = abs(metric_drift - 1.0) * surrogate_gain
    mode = str(split_plan.get("mode", "in_sample"))
    baseline_by_case = {str(r.get("case_id")): r for r in baseline_rows}
    cond_by_case = {str(c.get("case_id")): c for c in conditions}

    if surrogate_model_name != "linear_ridge":
        artifact = {"reference_rows": baseline_rows, "global_scale": metric_drift}
        surrogate_rows = run_surrogate_cases(artifact, conditions, qoi_cfg)
        _, summary = compare_with_baseline(baseline_rows, surrogate_rows, eval_cfg)
        return surrogate_rows, summary, []

    if mode == "in_sample":
        model = trained_surrogate or fit_lightweight_surrogate(
            conditions,
            baseline_rows,
            qoi_cfg,
            l2=surrogate_l2,
            split_cfg={"mode": "in_sample"},
        )
        artifact = {
            "linear_surrogate": model,
            "reference_rows": baseline_rows,
            "blend_reference": surrogate_blend,
            "perturb_scale": perturb_scale,
            "perturb_seed": seed + stage_idx,
        }
        surrogate_rows = run_surrogate_cases(artifact, conditions, qoi_cfg)
        _, summary = compare_with_baseline(baseline_rows, surrogate_rows, eval_cfg)
        return surrogate_rows, summary, []

    folds = list(split_plan.get("folds") or [])
    predicted: list[dict[str, Any]] = []
    baseline_eval: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    for fold_idx, test_case_ids in enumerate(folds):
        test_ids = [str(x) for x in test_case_ids]
        test_set = set(test_ids)
        train_conditions = [c for c in conditions if str(c.get("case_id")) not in test_set]
        train_baseline = [r for r in baseline_rows if str(r.get("case_id")) not in test_set]
        test_conditions = [cond_by_case[cid] for cid in test_ids if cid in cond_by_case]
        test_baseline = [baseline_by_case[cid] for cid in test_ids if cid in baseline_by_case]
        if not train_conditions or not test_conditions:
            continue

        model = fit_lightweight_surrogate(
            train_conditions,
            train_baseline,
            qoi_cfg,
            l2=surrogate_l2,
            split_cfg={"mode": "in_sample"},
        )
        artifact = {
            "linear_surrogate": model,
            "reference_rows": train_baseline,
            "blend_reference": surrogate_blend,
            "perturb_scale": perturb_scale,
            "perturb_seed": seed + stage_idx + fold_idx,
        }
        fold_pred = run_surrogate_cases(artifact, test_conditions, qoi_cfg)
        predicted.extend(fold_pred)
        baseline_eval.extend(test_baseline)
        _, fold_summary = compare_with_baseline(test_baseline, fold_pred, eval_cfg)
        fold_metrics.append(
            {
                "fold_index": int(fold_idx),
                "test_cases": [str(x) for x in test_ids],
                "n_test_cases": int(len(test_baseline)),
                "pass_rate": float(fold_summary.pass_rate),
                "mean_rel_diff": float(fold_summary.mean_rel_diff or 0.0),
                "max_rel_diff": float(fold_summary.max_rel_diff or 0.0),
                "qoi_metrics_count": int(fold_summary.qoi_metrics_count),
            }
        )

    if not predicted:
        artifact = {"reference_rows": baseline_rows, "global_scale": metric_drift}
        predicted = run_surrogate_cases(artifact, conditions, qoi_cfg)
        _, summary = compare_with_baseline(baseline_rows, predicted, eval_cfg)
        return predicted, summary, []

    _, summary = compare_with_baseline(baseline_eval, predicted, eval_cfg)
    return predicted, summary, fold_metrics


def _evaluate_physical_gate(
    *,
    enabled: bool,
    nu: np.ndarray,
    rop: np.ndarray,
    dt: np.ndarray,
    A: np.ndarray,
    X: np.ndarray,
    S: np.ndarray,
    keep_reactions: np.ndarray,
    max_conservation_violation: float,
    max_negative_steps: int,
    state_source: str,
    degraded: bool,
    fallback_reason: str | None,
) -> dict[str, Any]:
    nu_arr = np.asarray(nu, dtype=float)
    rop_arr = np.asarray(rop, dtype=float)
    dt_arr = np.asarray(dt, dtype=float)
    A_arr = np.asarray(A, dtype=float)
    x_arr = np.asarray(X, dtype=float)
    S_arr = np.asarray(S, dtype=float)
    keep = np.asarray(keep_reactions, dtype=bool)

    if dt_arr.shape != (rop_arr.shape[0],):
        raise ValueError("dt shape mismatch in physical gate evaluation")
    if x_arr.shape[0] != rop_arr.shape[0]:
        raise ValueError("X and rop row count mismatch in physical gate evaluation")
    if x_arr.shape[1] != nu_arr.shape[0]:
        raise ValueError("X species dimension mismatch in physical gate evaluation")

    if keep.shape != (nu_arr.shape[1],):
        raise ValueError("keep_reactions shape mismatch in physical gate evaluation")
    if not np.any(keep):
        keep = np.zeros((nu_arr.shape[1],), dtype=bool)
        keep[0] = True

    rop_keep = rop_arr[:, keep]
    nu_keep = nu_arr[:, keep]
    wdot_candidate = rop_keep @ nu_keep.T
    ydot_candidate = wdot_candidate @ S_arr

    y0 = x_arr[0] @ S_arr
    y_raw = np.zeros((x_arr.shape[0], S_arr.shape[1]), dtype=float)
    y_raw[0] = y0
    for i in range(1, y_raw.shape[0]):
        y_raw[i] = y_raw[i - 1] + ydot_candidate[i - 1] * float(max(dt_arr[i], 0.0))

    A_reduced = A_arr @ S_arr
    raw_conservation = float(conservation_violation(y_raw, A_reduced, reference=y0))
    raw_negative_steps = int(np.sum(np.any(y_raw < -1.0e-12, axis=1)))

    # Project candidate trajectory onto conservation manifold and nonnegative region
    # before gate judgment to avoid numerical-integration artifacts.
    y = np.asarray(
        project_to_conservation(
            y_raw,
            A_reduced,
            reference=y0,
            clip_nonnegative=True,
            max_iter=4,
        ),
        dtype=float,
    )
    conservation = float(conservation_violation(y, A_reduced, reference=y0))
    negative_steps = int(np.sum(np.any(y < -1.0e-12, axis=1)))
    passed = (
        conservation <= float(max_conservation_violation) + 1.0e-12
        and negative_steps <= int(max_negative_steps)
    )

    return {
        "enabled": bool(enabled),
        "passed": (passed if enabled else True),
        "raw_conservation_violation": raw_conservation,
        "raw_negative_steps": raw_negative_steps,
        "conservation_violation": conservation,
        "negative_steps": negative_steps,
        "max_conservation_violation": float(max_conservation_violation),
        "max_negative_steps": int(max_negative_steps),
        "state_source": state_source,
        "degraded": bool(degraded),
        "fallback_reason": fallback_reason,
        "trajectory_steps": int(y.shape[0]),
        "clusters": int(S_arr.shape[1]),
    }


def _formula_guess(name: str) -> dict[str, float]:
    out: dict[str, float] = {}
    token = ""
    num = ""

    def flush() -> None:
        nonlocal token, num
        if not token:
            return
        n = float(num) if num else 1.0
        out[token] = out.get(token, 0.0) + n
        token = ""
        num = ""

    for ch in name:
        if ch.isupper():
            flush()
            token = ch
        elif ch.islower() and token:
            token += ch
        elif ch.isdigit() and token:
            num += ch
        else:
            flush()
    flush()
    return out


def _guess_species_meta(species_names: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in species_names:
        out.append(
            {
                "name": name,
                "composition": _formula_guess(name),
                "phase": "gas",
                "charge": 0,
                "radical": False,
                "role": "",
            }
        )
    return out


def _ignition_delay(time: np.ndarray, temp: np.ndarray) -> float:
    if time.size < 3:
        return float("nan")
    dt = np.diff(time)
    dT = np.diff(temp)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(dt > 0.0, dT / dt, -np.inf)
    idx = int(np.argmax(slope))
    return float(time[idx + 1])


def _qoi_from_trace_case(case: Any, qoi_cfg: dict[str, Any]) -> dict[str, float]:
    species_last = list(qoi_cfg.get("species_last") or [])
    species_max = list(qoi_cfg.get("species_max") or [])
    species_integral = list(qoi_cfg.get("species_integral") or [])
    deposition_integral = list(qoi_cfg.get("deposition_integral") or [])

    idx = {s: i for i, s in enumerate(case.species_names)}
    time = np.asarray(case.time, dtype=float)
    out: dict[str, float] = {
        "ignition_delay": _ignition_delay(time, case.temperature),
        "T_max": float(np.max(case.temperature)),
        "T_last": float(case.temperature[-1]),
    }
    for sp in species_last:
        i = idx.get(sp)
        out[f"X_last:{sp}"] = float(max(0.0, case.X[-1, i])) if i is not None else float("nan")
    for sp in species_max:
        i = idx.get(sp)
        out[f"X_max:{sp}"] = float(np.max(np.maximum(case.X[:, i], 0.0))) if i is not None else float("nan")
    for sp in species_integral:
        i = idx.get(sp)
        if i is None:
            out[f"X_int:{sp}"] = float("nan")
            continue
        series = np.maximum(np.asarray(case.X[:, i], dtype=float), 0.0)
        out[f"X_int:{sp}"] = (
            float(np.trapezoid(series, time)) if time.size > 1 else float(series[-1]) if series.size else 0.0
        )
    for sp in deposition_integral:
        i = idx.get(sp)
        if i is None:
            out[f"dep_int:{sp}"] = float("nan")
            continue
        series = np.maximum(np.asarray(case.X[:, i], dtype=float), 0.0)
        out[f"dep_int:{sp}"] = (
            float(np.trapezoid(series, time)) if time.size > 1 else float(series[-1]) if series.size else 0.0
        )
    return out


def _reorder_rows(source_names: list[str], target_names: list[str], matrix: np.ndarray) -> np.ndarray:
    source_idx = {name: i for i, name in enumerate(source_names)}
    out = np.zeros((len(target_names), matrix.shape[1]), dtype=float)
    for t_idx, name in enumerate(target_names):
        s_idx = source_idx.get(name)
        if s_idx is not None:
            out[t_idx, :] = matrix[s_idx, :]
    return out


def _case_dt(time: np.ndarray) -> np.ndarray:
    t = np.asarray(time, dtype=float)
    if t.size < 2:
        return np.ones_like(t)
    dt = np.empty_like(t)
    dt[0] = max(float(t[1] - t[0]), 1.0e-12)
    dt[1:] = np.maximum(np.diff(t), 1.0e-12)
    return dt


def _bundle_state_arrays(bundle: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    times: list[np.ndarray] = []
    xs: list[np.ndarray] = []
    dts: list[np.ndarray] = []
    case_slices: list[dict[str, Any]] = []
    cursor = 0
    for case in bundle.cases:
        t = np.asarray(case.time, dtype=float)
        x = np.asarray(case.X, dtype=float)
        dt = _case_dt(t)
        start = cursor
        end = cursor + int(t.shape[0])
        case_slices.append(
            {
                "case_id": str(case.case_id),
                "start": start,
                "end": end,
                "n_steps": int(t.shape[0]),
            }
        )
        cursor = end
        times.append(t)
        xs.append(x)
        dts.append(dt)

    time_concat = np.concatenate(times, axis=0) if times else np.zeros((0,), dtype=float)
    x_concat = np.concatenate(xs, axis=0) if xs else np.zeros((0, len(bundle.species_names)), dtype=float)
    dt_concat = np.concatenate(dts, axis=0) if dts else np.zeros((0,), dtype=float)
    return (
        np.asarray(time_concat, dtype=float),
        np.asarray(x_concat, dtype=float),
        np.asarray(dt_concat, dtype=float),
        case_slices,
    )


def _case_feature_vector(case: dict[str, Any]) -> np.ndarray:
    T0 = float(case.get("T0", 1000.0))
    P0 = float(case.get("P0_atm", 1.0))
    phi = float(case.get("phi", 1.0))
    t_end = float(case.get("t_end", 0.1))
    return np.asarray(
        [
            T0 / 1500.0,
            P0 / 2.0,
            phi,
            t_end,
        ],
        dtype=float,
    )


def _build_learnckpp_features(
    *,
    time: np.ndarray,
    case_slices: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
) -> np.ndarray:
    t = np.asarray(time, dtype=float)
    if t.ndim != 1:
        raise ValueError("time must be 1-D for learnckpp feature construction")
    if t.size == 0:
        return np.zeros((0, 6), dtype=float)

    features = np.zeros((t.size, 6), dtype=float)
    global_span = max(float(t[-1] - t[0]), 1.0e-12)
    global_t = (t - float(t[0])) / global_span
    features[:, 0] = 1.0
    features[:, 1] = global_t
    features[:, 2] = global_t * global_t

    cond_by_case = {str(c.get("case_id")): c for c in conditions}
    default_cond = conditions[0] if conditions else {"T0": 1000.0, "P0_atm": 1.0, "phi": 1.0, "t_end": float(t[-1])}

    if not case_slices:
        vec = _case_feature_vector(default_cond)
        features[:, 3:] = vec[None, :3]
        return features

    for sl in case_slices:
        start = int(sl.get("start", 0))
        end = int(sl.get("end", 0))
        if not (0 <= start < end <= t.size):
            continue
        case_id = str(sl.get("case_id", ""))
        cond = cond_by_case.get(case_id, default_cond)
        vec = _case_feature_vector(cond)
        features[start:end, 3:] = vec[None, :3]

    fill_mask = np.all(np.isclose(features[:, 3:], 0.0), axis=1)
    if np.any(fill_mask):
        vec = _case_feature_vector(default_cond)
        features[fill_mask, 3:] = vec[None, :3]

    return features


def _resolve_learnckpp_target_keep_ratio(
    *,
    base_keep_ratio: float,
    prune_keep_ratio: float,
    stage_idx: int,
    data_source: str,
    split_mode: str,
    max_mean_rel: float,
    prev_mean_rel_diff: float | None,
    prev_stage_physical: dict[str, Any] | None,
    adaptive_cfg: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    raw = float(base_keep_ratio) * float(prune_keep_ratio)
    raw = float(np.clip(raw, 1.0e-6, 1.0))

    enabled = bool(adaptive_cfg.get("enabled", True))
    if not enabled:
        return raw, {"enabled": False, "raw": raw, "adjusted": raw}

    min_keep = float(adaptive_cfg.get("min_keep_ratio", 0.10))
    max_keep = float(adaptive_cfg.get("max_keep_ratio", 0.95))
    source_mult_map = dict(adaptive_cfg.get("source_multiplier") or {})
    split_mult_map = dict(adaptive_cfg.get("split_multiplier") or {})
    stage_mult = np.asarray(adaptive_cfg.get("stage_multiplier") or [1.0, 1.0, 1.0], dtype=float)
    if stage_mult.size == 0:
        stage_mult = np.asarray([1.0], dtype=float)

    source_mult = float(source_mult_map.get(data_source, source_mult_map.get("default", 1.0)))
    split_mult = float(split_mult_map.get(split_mode, split_mult_map.get("default", 1.0)))
    stage_mult_val = float(stage_mult[min(stage_idx, int(stage_mult.size) - 1)])

    rel_feedback = float(adaptive_cfg.get("feedback_multiplier", 1.15))
    rel_trigger = float(adaptive_cfg.get("feedback_trigger_ratio", 0.80))
    feedback_mult = 1.0
    if prev_mean_rel_diff is not None and max_mean_rel > 0.0:
        if float(prev_mean_rel_diff) >= rel_trigger * max_mean_rel:
            feedback_mult = rel_feedback

    physical_cfg = dict(adaptive_cfg.get("physical_feedback") or {})
    physical_enabled = bool(physical_cfg.get("enabled", True))
    physical_mult = 1.0
    physical_reason: list[str] = []
    if physical_enabled and prev_stage_physical:
        prev_passed = bool(prev_stage_physical.get("passed", True))
        raw_cons = float(prev_stage_physical.get("raw_conservation_violation", 0.0))
        raw_neg = int(prev_stage_physical.get("raw_negative_steps", 0))

        if not prev_passed:
            physical_mult = max(physical_mult, float(physical_cfg.get("fail_multiplier", 1.25)))
            physical_reason.append("prev_stage_failed")
        cons_th = float(physical_cfg.get("raw_conservation_threshold", 1.0))
        if raw_cons > cons_th:
            physical_mult = max(physical_mult, float(physical_cfg.get("cons_multiplier", 1.10)))
            physical_reason.append("high_raw_conservation")
        neg_th = int(physical_cfg.get("raw_negative_steps_threshold", 8))
        if raw_neg > neg_th:
            physical_mult = max(physical_mult, float(physical_cfg.get("negative_multiplier", 1.10)))
            physical_reason.append("high_raw_negative")

    adjusted = raw * source_mult * split_mult * stage_mult_val * feedback_mult * physical_mult
    adjusted = float(np.clip(adjusted, min_keep, max_keep))
    return adjusted, {
        "enabled": True,
        "raw": raw,
        "adjusted": adjusted,
        "source_multiplier": source_mult,
        "split_multiplier": split_mult,
        "stage_multiplier": stage_mult_val,
        "feedback_multiplier": feedback_mult,
        "feedback_trigger_ratio": rel_trigger,
        "physical_feedback_enabled": physical_enabled,
        "physical_feedback_multiplier": physical_mult,
        "physical_feedback_reason": physical_reason,
        "min_keep_ratio": min_keep,
        "max_keep_ratio": max_keep,
    }


def _auto_tune_learnckpp_keep_ratio(
    *,
    base_keep_ratio: float,
    overall_candidates: int,
    min_keep_ratio: float,
    max_keep_ratio: float,
    data_source: str,
    split_mode: str,
    prev_stage_physical: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return float(base_keep_ratio), {"enabled": False, "selected": float(base_keep_ratio)}

    n_cand = int(max(overall_candidates, 0))
    if n_cand <= 0:
        return float(base_keep_ratio), {"enabled": True, "selected": float(base_keep_ratio), "reason": "empty_candidates"}

    multipliers_raw = cfg.get("multipliers") or [0.90, 1.00, 1.10, 1.25]
    multipliers: list[float] = []
    for val in multipliers_raw:
        try:
            multipliers.append(float(val))
        except (TypeError, ValueError):
            continue
    if not multipliers:
        multipliers = [1.0]

    compression_weight = float(cfg.get("compression_weight", 1.0))
    safety_weight = float(cfg.get("safety_weight", 2.0))
    min_keep_floor = float(cfg.get("min_keep_floor", 0.10))
    risk_keep_boost = float(cfg.get("risk_keep_boost", 0.35))
    risk_cap = float(cfg.get("risk_cap", 3.0))

    source_risk_map = dict(cfg.get("source_risk") or {})
    split_risk_map = dict(cfg.get("split_risk") or {})
    risk = float(source_risk_map.get(data_source, source_risk_map.get("default", 0.0)))
    risk += float(split_risk_map.get(split_mode, split_risk_map.get("default", 0.0)))

    phys_cfg = dict(cfg.get("physical_risk") or {})
    if prev_stage_physical:
        raw_cons = float(prev_stage_physical.get("raw_conservation_violation", 0.0))
        raw_neg = float(prev_stage_physical.get("raw_negative_steps", 0.0))
        passed = bool(prev_stage_physical.get("passed", True))
        cons_norm = float(max(phys_cfg.get("cons_norm", 1.0), 1.0e-12))
        neg_norm = float(max(phys_cfg.get("neg_norm", 20.0), 1.0))
        cons_w = float(phys_cfg.get("cons_weight", 1.0))
        neg_w = float(phys_cfg.get("neg_weight", 1.0))
        fail_penalty = float(phys_cfg.get("fail_penalty", 1.0))

        risk += cons_w * min(raw_cons / cons_norm, risk_cap)
        risk += neg_w * min(raw_neg / neg_norm, risk_cap)
        if not passed:
            risk += fail_penalty

    risk = float(np.clip(risk, 0.0, risk_cap))
    desired_min_keep = float(np.clip(min_keep_floor + risk_keep_boost * (risk / max(risk_cap, 1.0e-12)), min_keep_ratio, max_keep_ratio))

    candidates_meta: list[dict[str, Any]] = []
    best_ratio = float(base_keep_ratio)
    best_score = -float("inf")
    best_keep_ratio = 1.0
    for mult in multipliers:
        ratio = float(np.clip(base_keep_ratio * mult, min_keep_ratio, max_keep_ratio))
        keep_count = max(1, min(n_cand, int(round(ratio * n_cand))))
        keep_ratio = float(keep_count / n_cand)
        compression_gain = 1.0 - keep_ratio
        safety_penalty = max(0.0, desired_min_keep - keep_ratio)
        score = compression_weight * compression_gain - safety_weight * safety_penalty
        candidates_meta.append(
            {
                "multiplier": float(mult),
                "ratio": ratio,
                "keep_ratio": keep_ratio,
                "score": float(score),
                "safety_penalty": float(safety_penalty),
            }
        )
        if score > best_score + 1.0e-12:
            best_score = float(score)
            best_ratio = ratio
            best_keep_ratio = keep_ratio
        elif abs(score - best_score) <= 1.0e-12:
            # Risky regime prefers higher keep ratio; safe regime prefers lower keep ratio.
            if risk >= 1.0 and keep_ratio > best_keep_ratio:
                best_ratio = ratio
                best_keep_ratio = keep_ratio
            if risk < 1.0 and keep_ratio < best_keep_ratio:
                best_ratio = ratio
                best_keep_ratio = keep_ratio

    return best_ratio, {
        "enabled": True,
        "selected": float(best_ratio),
        "base": float(base_keep_ratio),
        "risk": float(risk),
        "desired_min_keep": float(desired_min_keep),
        "candidates": candidates_meta,
    }


def _fit_pooling_mapping(
    *,
    nu: np.ndarray,
    rop: np.ndarray,
    F_bar: np.ndarray | None,
    X: np.ndarray,
    wdot: np.ndarray,
    species_meta: list[dict[str, Any]],
    target_ratio: float,
    pooling_cfg: dict[str, Any],
    artifact_path: Path | None,
) -> tuple[ReductionMapping, dict[str, Any], Path | None]:
    nu_arr = np.asarray(nu, dtype=float)
    rop_arr = np.asarray(rop, dtype=float)
    x_arr = np.asarray(X, dtype=float)
    wdot_arr = np.asarray(wdot, dtype=float)
    f_arr = None if F_bar is None else np.asarray(F_bar, dtype=float)

    cfg = dict(pooling_cfg or {})
    train_cfg = dict(cfg.get("train") or {})
    train_cfg["target_ratio"] = float(target_ratio)
    cfg["train"] = train_cfg

    graph_kind = str(cfg.get("graph", "species"))
    rop_stats = np.mean(np.abs(rop_arr), axis=0) if rop_arr.size else np.zeros((nu_arr.shape[1],), dtype=float)
    if graph_kind == "bipartite":
        graph = build_bipartite_graph(nu_arr, rop_stats, species_meta, cfg.get("graph_cfg"))
    else:
        graph = build_species_graph(nu_arr, f_arr, species_meta, cfg.get("graph_cfg"))

    phase_labels = [str(m.get("phase", "")) for m in species_meta]
    features = extract_species_features(species_meta, x_arr, wdot_arr, phase_labels, cfg.get("feature_cfg"))
    hard_mask = build_hard_mask(species_meta, cfg.get("constraint_cfg"))
    pair_cost = build_pairwise_cost(species_meta, cfg.get("constraint_cfg"))
    constraints = {
        "hard_mask": hard_mask,
        "pair_cost": pair_cost,
        "species_meta": species_meta,
    }
    trained = train_pooling_assignment(graph, features, constraints, cfg)
    S = np.asarray(trained.get("S"), dtype=float)
    cluster_meta = list(trained.get("cluster_meta") or [])
    train_metrics = dict(trained.get("train_metrics") or {})
    hard_viol = int(train_metrics.get("hard_ban_violations", 0))

    mapping = ReductionMapping(
        S=S,
        pool_meta=cluster_meta,
        keep_reactions=None,
        meta={
            "target_ratio": float(target_ratio),
            "achieved_ratio": float(S.shape[1]) / float(max(S.shape[0], 1)),
            "hard_ban_violations": int(hard_viol),
            "pooling_model": str((trained.get("model_info") or {}).get("model_type", "unknown")),
            "pooling_graph": str((trained.get("model_info") or {}).get("graph_type", graph_kind)),
        },
    )

    saved_path: Path | None = None
    if artifact_path is not None:
        saved_path = save_pooling_artifact(
            artifact_path,
            {
                "S": S,
                "S_prob": np.asarray(trained.get("S_prob"), dtype=float),
                "cluster_meta": cluster_meta,
                "train_metrics": train_metrics,
                "model_info": dict(trained.get("model_info") or {}),
            },
        )

    return mapping, {
        "train_metrics": train_metrics,
        "model_info": dict(trained.get("model_info") or {}),
        "graph_kind": graph_kind,
        "hard_ban_violations": int(hard_viol),
    }, saved_path


def _cluster_preview(
    *,
    S: np.ndarray,
    pool_meta: list[dict[str, Any]] | None,
    species_meta: list[dict[str, Any]],
    max_clusters: int = 8,
    max_members: int = 6,
) -> list[dict[str, Any]]:
    if pool_meta:
        rows: list[dict[str, Any]] = []
        for entry in pool_meta:
            members = [str(x) for x in list(entry.get("members") or [])]
            rows.append(
                {
                    "cluster_id": int(entry.get("cluster_id", len(rows))),
                    "size": int(len(members)),
                    "members_sample": members[:max_members],
                    "elements": [str(x) for x in list(entry.get("elements") or [])],
                }
            )
        rows.sort(key=lambda r: (-int(r["size"]), int(r["cluster_id"])))
        return rows[:max_clusters]

    s_arr = np.asarray(S, dtype=float)
    if s_arr.ndim != 2:
        return []
    names = [str(m.get("name", f"sp{i}")) for i, m in enumerate(species_meta)]
    out: list[dict[str, Any]] = []
    for c_idx in range(s_arr.shape[1]):
        members = np.where(s_arr[:, c_idx] > 0.5)[0].tolist()
        member_names = [names[i] for i in members]
        elem_union: set[str] = set()
        for i in members:
            comp = species_meta[i].get("composition") or {}
            elem_union.update(str(k) for k in comp.keys())
        out.append(
            {
                "cluster_id": int(c_idx),
                "size": int(len(members)),
                "members_sample": member_names[:max_members],
                "elements": sorted(elem_union),
            }
        )
    out.sort(key=lambda r: (-int(r["size"]), int(r["cluster_id"])))
    return out[:max_clusters]


def _reconstruct_state_from_wdot(wdot: np.ndarray, dt: np.ndarray) -> np.ndarray:
    arr = np.asarray(wdot, dtype=float)
    step = np.asarray(dt, dtype=float)
    if arr.ndim != 2:
        raise ValueError("wdot must be 2-D")
    if step.shape != (arr.shape[0],):
        raise ValueError("dt shape mismatch with wdot rows")
    out = np.zeros_like(arr)
    if arr.shape[0] == 0:
        return out
    for i in range(1, arr.shape[0]):
        out[i] = out[i - 1] + arr[i - 1] * step[i]
    min_vals = np.min(out, axis=0, keepdims=True)
    out = out - np.minimum(min_vals, 0.0)
    return out


def _load_trace_context(cfg: dict[str, Any], *, config_parent: Path) -> dict[str, Any] | None:
    trace_h5 = cfg.get("trace_h5")
    if not trace_h5:
        return None

    bundle = load_case_bundle(_resolve_path(str(trace_h5), base=config_parent))
    if not bundle.cases:
        raise ValueError("trace_h5 has no cases")

    rop = np.concatenate([c.rop for c in bundle.cases], axis=0)
    wdot = np.concatenate([c.wdot for c in bundle.cases], axis=0)

    nu: np.ndarray
    nu_meta = (bundle.meta or {}).get("nu")
    if nu_meta is not None:
        nu = np.asarray(nu_meta, dtype=float)
    else:
        try:
            nu_sparse, mech_species, _ = build_nu(bundle.mechanism_path, bundle.phase)
            nu = nu_sparse.toarray()
            if list(mech_species) != list(bundle.species_names):
                nu = _reorder_rows(list(mech_species), list(bundle.species_names), nu)
        except Exception:
            # fallback when cantera is unavailable: infer linear mapping wdot ~= rop @ nu.T
            nu = np.linalg.lstsq(rop, wdot, rcond=None)[0].T

    if nu.shape != (len(bundle.species_names), len(bundle.reaction_eqs)):
        raise ValueError("nu shape mismatch with trace species/reactions")

    merge_cfg = cfg.get("merge") or {}
    species_meta = list(merge_cfg.get("species_meta") or [])
    if not species_meta:
        species_meta = list((bundle.meta or {}).get("species_meta") or [])
    if not species_meta:
        try:
            species_meta = extract_species_meta(bundle.mechanism_path, phase=bundle.phase)
        except Exception:
            species_meta = _guess_species_meta(bundle.species_names)

    if len(species_meta) != len(bundle.species_names):
        by_name = {str(m.get("name", "")): dict(m) for m in species_meta}
        species_meta = [by_name.get(name, {"name": name, "composition": _formula_guess(name), "phase": bundle.phase, "charge": 0, "radical": False, "role": ""}) for name in bundle.species_names]

    f_cfg = (cfg.get("merge") or {}).get("F_bar")
    if f_cfg is not None:
        F_bar = np.asarray(f_cfg, dtype=float)
    else:
        F_bar = build_flux(nu, rop)
    i_reaction = reaction_importance(rop)
    time, X, dt, case_slices = _bundle_state_arrays(bundle)

    conditions: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    qoi_cfg = dict(cfg.get("qoi") or {})
    for case in bundle.cases:
        cond = dict((case.meta or {}).get("conditions") or {})
        cond.setdefault("case_id", case.case_id)
        cond.setdefault("T0", float(case.temperature[0]))
        cond.setdefault("P0_atm", float(case.pressure[0]) / 101325.0)
        cond.setdefault("phi", 1.0)
        cond.setdefault("t_end", float(case.time[-1]))
        conditions.append(cond)

        baseline_rows.append({"case_id": case.case_id, **_qoi_from_trace_case(case, qoi_cfg)})

    return {
        "nu": nu,
        "rop": rop,
        "wdot": wdot,
        "time": time,
        "X": X,
        "dt": dt,
        "case_slices": case_slices,
        "species_meta": species_meta,
        "species_names": list(bundle.species_names),
        "reaction_eqs": list(bundle.reaction_eqs),
        "F_bar": F_bar,
        "i_reaction": i_reaction,
        "conditions": conditions,
        "baseline_rows": baseline_rows,
    }


def _load_bundle_conditions_and_baseline(
    *,
    trace_h5: Path,
    qoi_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bundle = load_case_bundle(trace_h5)
    if not bundle.cases:
        raise ValueError(f"trace_h5 has no cases: {trace_h5}")

    conditions: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for case in bundle.cases:
        cond = dict((case.meta or {}).get("conditions") or {})
        cond.setdefault("case_id", case.case_id)
        cond.setdefault("T0", float(case.temperature[0]))
        cond.setdefault("P0_atm", float(case.pressure[0]) / 101325.0)
        cond.setdefault("phi", 1.0)
        cond.setdefault("t_end", float(case.time[-1]))
        conditions.append(cond)
        baseline_rows.append({"case_id": case.case_id, **_qoi_from_trace_case(case, qoi_cfg)})

    return conditions, baseline_rows


def _parse_phase_weights(raw: Any, phase_names: list[str]) -> np.ndarray:
    n_phase = len(phase_names)
    if n_phase < 1:
        raise ValueError("phase_names must not be empty")

    if isinstance(raw, dict):
        vals = [float(raw.get(name, 0.0)) for name in phase_names]
    elif isinstance(raw, (list, tuple)):
        if len(raw) != n_phase:
            raise ValueError(f"phase_weights list length must be {n_phase}")
        vals = [float(v) for v in raw]
    elif isinstance(raw, str):
        parsed: dict[str, float] = {}
        for token in raw.split(","):
            part = token.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError(f"invalid phase_weights token: {part!r}")
            name, value = part.split(":", 1)
            parsed[name.strip()] = float(value.strip())
        vals = [float(parsed.get(name, 0.0)) for name in phase_names]
    else:
        raise ValueError("merge.phase_weights must be dict/list/string")

    arr = np.asarray(vals, dtype=float)
    arr = np.maximum(arr, 0.0)
    total = float(np.sum(arr))
    if total <= 0.0:
        raise ValueError("phase_weights must contain at least one positive weight")
    return arr / total


def _phase_index_from_select(phase_select: Any, phase_names: list[str]) -> int:
    if isinstance(phase_select, int):
        idx = int(phase_select)
    elif isinstance(phase_select, str):
        text = phase_select.strip()
        if text in phase_names:
            idx = phase_names.index(text)
        elif text.isdigit():
            idx = int(text)
        else:
            raise ValueError(f"unknown phase_select value: {phase_select!r}")
    else:
        raise ValueError("merge.phase_select must be int or str")

    if not (0 <= idx < len(phase_names)):
        raise ValueError(f"phase_select index out of range: {idx}")
    return idx


def _resolve_phase_context(
    cfg: dict[str, Any],
    *,
    network_dir: Path,
    F_bar_default: np.ndarray,
    i_reaction_default: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    merge_cfg = dict(cfg.get("merge") or {})
    phase_select = merge_cfg.get("phase_select")
    phase_weights_raw = merge_cfg.get("phase_weights")
    if phase_select is not None and phase_weights_raw is not None:
        raise ValueError("merge.phase_select and merge.phase_weights cannot be set at the same time")

    if phase_select is None and phase_weights_raw is None:
        return (
            F_bar_default,
            i_reaction_default,
            {
                "mode": "global",
                "applied": False,
                "phase_names": [],
                "weights": [],
                "n_phase": 0,
                "selected_phase": None,
                "used_i_reaction_by_phase": False,
                "source": "F_bar.npy",
            },
        )

    f_phase_path = network_dir / "F_bar_by_phase.npy"
    i_phase_path = network_dir / "I_reaction_by_phase.npy"
    phase_names_path = network_dir / "phase_names.json"
    if not f_phase_path.exists():
        raise FileNotFoundError(
            "merge.phase_select/phase_weights was set, but F_bar_by_phase.npy is missing in network_dir"
        )

    F_by_phase = np.asarray(np.load(f_phase_path), dtype=float)
    if F_by_phase.ndim != 3:
        raise ValueError("F_bar_by_phase.npy must be 3-D (Nphase, Ns, Ns)")
    if F_by_phase.shape[1:] != F_bar_default.shape:
        raise ValueError("F_bar_by_phase shape mismatch with F_bar.npy")

    if phase_names_path.exists():
        phase_names = [str(x) for x in json.loads(phase_names_path.read_text())]
    else:
        phase_names = [f"phase_{i}" for i in range(F_by_phase.shape[0])]
    if len(phase_names) != F_by_phase.shape[0]:
        raise ValueError("phase_names length mismatch with F_bar_by_phase first dimension")

    I_by_phase: np.ndarray | None = None
    if i_phase_path.exists():
        I_by_phase = np.asarray(np.load(i_phase_path), dtype=float)
        if I_by_phase.shape != (F_by_phase.shape[0], i_reaction_default.shape[0]):
            raise ValueError("I_reaction_by_phase shape mismatch")

    if phase_select is not None:
        idx = _phase_index_from_select(phase_select, phase_names)
        weights = np.zeros(len(phase_names), dtype=float)
        weights[idx] = 1.0
        mode = "select"
        selected_phase = phase_names[idx]
    else:
        weights = _parse_phase_weights(phase_weights_raw, phase_names)
        mode = "weights"
        selected_phase = None

    F_bar = np.tensordot(weights, F_by_phase, axes=(0, 0))
    i_reaction = (
        np.tensordot(weights, I_by_phase, axes=(0, 0))
        if I_by_phase is not None
        else i_reaction_default
    )

    return (
        np.asarray(F_bar, dtype=float),
        np.asarray(i_reaction, dtype=float),
        {
            "mode": mode,
            "applied": True,
            "phase_names": phase_names,
            "weights": weights.tolist(),
            "n_phase": len(phase_names),
            "selected_phase": selected_phase,
            "used_i_reaction_by_phase": bool(I_by_phase is not None),
            "source": "F_bar_by_phase.npy",
        },
    )


def _load_network_context(
    cfg: dict[str, Any],
    *,
    config_parent: Path,
    qoi_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    network_dir_raw = cfg.get("network_dir")
    if not network_dir_raw:
        return None

    network_dir = _resolve_path(str(network_dir_raw), base=config_parent)
    if not network_dir.exists():
        raise FileNotFoundError(f"network_dir not found: {network_dir}")
    if not network_dir.is_dir():
        raise ValueError(f"network_dir is not a directory: {network_dir}")

    nu_path = network_dir / "nu.npy"
    rop_path = network_dir / "rop.npy"
    wdot_path = network_dir / "wdot.npy"
    dt_path = network_dir / "dt.npy"
    time_path = network_dir / "time.npy"
    x_path = network_dir / "X.npy"
    case_slices_path = network_dir / "case_slices.json"
    fbar_path = network_dir / "F_bar.npy"
    ireact_path = network_dir / "I_reaction.npy"
    species_meta_path = network_dir / "species_meta.json"
    conditions_path = network_dir / "conditions.json"
    summary_path = network_dir / "summary.json"

    required = [nu_path, rop_path, wdot_path, fbar_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"network_dir missing required artifacts: {missing}")

    nu = np.load(nu_path)
    rop = np.load(rop_path)
    wdot = np.load(wdot_path)
    dt = np.load(dt_path) if dt_path.exists() else np.ones((rop.shape[0],), dtype=float)
    F_bar = np.load(fbar_path)
    i_reaction = np.load(ireact_path) if ireact_path.exists() else reaction_importance(rop)
    F_bar, i_reaction, phase_context = _resolve_phase_context(
        cfg,
        network_dir=network_dir,
        F_bar_default=np.asarray(F_bar, dtype=float),
        i_reaction_default=np.asarray(i_reaction, dtype=float),
    )

    if species_meta_path.exists():
        species_meta = list(json.loads(species_meta_path.read_text()))
    else:
        species_names_path = network_dir / "species_names.json"
        if species_names_path.exists():
            species_names = list(json.loads(species_names_path.read_text()))
            species_meta = _guess_species_meta(species_names)
        else:
            species_meta = _default_species_meta()

    conditions: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    if conditions_path.exists():
        conditions = list(json.loads(conditions_path.read_text()))

    trace_h5_candidate: Path | None = None
    if summary_path.exists():
        summary = dict(json.loads(summary_path.read_text()))
        trace_h5_raw = summary.get("trace_h5")
        if trace_h5_raw:
            trace_h5_candidate = _resolve_path(str(trace_h5_raw), base=network_dir)

    if trace_h5_candidate is not None and trace_h5_candidate.exists():
        conditions, baseline_rows = _load_bundle_conditions_and_baseline(
            trace_h5=trace_h5_candidate,
            qoi_cfg=qoi_cfg,
        )
    elif conditions:
        baseline_rows = run_surrogate_cases({"global_scale": 1.0}, conditions, qoi_cfg)
    else:
        conditions_csv = cfg.get("conditions_csv")
        if not conditions_csv:
            raise ValueError("network_dir mode requires trace_h5 reference or conditions source")
        conditions = load_conditions(_resolve_path(str(conditions_csv), base=config_parent))
        baseline_rows = run_surrogate_cases({"global_scale": 1.0}, conditions, qoi_cfg)

    state_source = "network_artifacts"
    degraded = False
    fallback_reason: str | None = None

    time: np.ndarray
    X: np.ndarray
    case_slices: list[dict[str, Any]]
    if time_path.exists() and x_path.exists():
        time = np.asarray(np.load(time_path), dtype=float)
        X = np.asarray(np.load(x_path), dtype=float)
        if X.ndim != 2 or time.ndim != 1 or X.shape[0] != time.shape[0]:
            raise ValueError("network_dir time/X shape mismatch")
        if case_slices_path.exists():
            case_slices = list(json.loads(case_slices_path.read_text()))
        else:
            case_slices = []
    elif trace_h5_candidate is not None and trace_h5_candidate.exists():
        bundle = load_case_bundle(trace_h5_candidate)
        time, X, dt_bundle, case_slices = _bundle_state_arrays(bundle)
        if dt.shape != dt_bundle.shape:
            dt = dt_bundle
        state_source = "trace_h5_fallback"
        degraded = True
        fallback_reason = "state_artifacts_missing_used_trace_h5"
    else:
        time = np.cumsum(np.asarray(dt, dtype=float))
        X = _reconstruct_state_from_wdot(wdot, np.asarray(dt, dtype=float))
        case_slices = [
            {
                "case_id": "reconstructed",
                "start": 0,
                "end": int(X.shape[0]),
                "n_steps": int(X.shape[0]),
            }
        ]
        state_source = "wdot_reconstructed"
        degraded = True
        fallback_reason = "state_artifacts_missing_reconstructed_from_wdot"

    return {
        "nu": nu,
        "rop": rop,
        "wdot": wdot,
        "dt": np.asarray(dt, dtype=float),
        "time": np.asarray(time, dtype=float),
        "X": np.asarray(X, dtype=float),
        "case_slices": case_slices,
        "state_source": state_source,
        "state_degraded": degraded,
        "state_fallback_reason": fallback_reason,
        "species_meta": species_meta,
        "F_bar": F_bar,
        "i_reaction": i_reaction,
        "conditions": conditions,
        "baseline_rows": baseline_rows,
        "phase_context": phase_context,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggressive merge + prune + surrogate validation")
    parser.add_argument("--config", required=True, help="Path to reduce/validate config")
    parser.add_argument("--run-id", required=True, help="Run identifier")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)

    qoi_cfg = dict(cfg.get("qoi") or {"species_last": ["CO2", "CO", "CH4", "O2"], "species_max": ["OH", "HO2"]})
    qoi_species_integral = [str(x) for x in list(qoi_cfg.get("species_integral") or []) if str(x)]
    qoi_deposition_integral = [str(x) for x in list(qoi_cfg.get("deposition_integral") or []) if str(x)]
    qoi_integral_keys = [f"X_int:{sp}" for sp in qoi_species_integral] + [f"dep_int:{sp}" for sp in qoi_deposition_integral]
    qoi_integral_count = int(len(qoi_integral_keys))
    eval_cfg = dict(cfg.get("evaluation") or {"rel_tolerance": 0.4, "rel_eps": 1.0e-12})

    network_ctx = _load_network_context(cfg, config_parent=config_path.parent, qoi_cfg=qoi_cfg)
    trace_ctx = None if network_ctx is not None else _load_trace_context(cfg, config_parent=config_path.parent)
    data_source = "network_dir" if network_ctx is not None else ("trace_h5" if trace_ctx is not None else "synthetic")

    if network_ctx is not None:
        conditions = network_ctx["conditions"]
        baseline_rows = network_ctx["baseline_rows"]
        species_meta = network_ctx["species_meta"]
        nu = np.asarray(network_ctx["nu"], dtype=float)
        rop = np.asarray(network_ctx["rop"], dtype=float)
        wdot = np.asarray(network_ctx["wdot"], dtype=float)
        dt = np.asarray(network_ctx["dt"], dtype=float)
        time = np.asarray(network_ctx["time"], dtype=float)
        X = np.asarray(network_ctx["X"], dtype=float)
        case_slices = list(network_ctx.get("case_slices") or [])
        F_bar = np.asarray(network_ctx["F_bar"], dtype=float)
        i_reaction = np.asarray(network_ctx["i_reaction"], dtype=float)
        state_source = str(network_ctx.get("state_source", "network_artifacts"))
        state_degraded = bool(network_ctx.get("state_degraded", False))
        state_fallback_reason = network_ctx.get("state_fallback_reason")
        n_species = nu.shape[0]
        n_reactions = nu.shape[1]
        seed = int((cfg.get("search") or {}).get("seed", 7))
        phase_context = dict(
            network_ctx.get("phase_context")
            or {
                "mode": "global",
                "applied": False,
                "phase_names": [],
                "weights": [],
                "n_phase": 0,
                "selected_phase": None,
                "used_i_reaction_by_phase": False,
                "source": "F_bar.npy",
            }
        )
    elif trace_ctx is None:
        conditions_csv = cfg.get("conditions_csv", "assets/conditions/gri30_tiny.csv")
        conditions = load_conditions(_resolve_path(str(conditions_csv), base=config_path.parent))
        baseline_rows = run_surrogate_cases({"global_scale": 1.0}, conditions, qoi_cfg)

        species_meta = list((cfg.get("merge") or {}).get("species_meta") or _default_species_meta())
        n_species = len(species_meta)
        n_reactions = int(((cfg.get("synthetic") or {}).get("n_reactions", 24)))
        seed = int((cfg.get("search") or {}).get("seed", 7))

        F_bar = np.asarray(((cfg.get("merge") or {}).get("F_bar") or np.zeros((n_species, n_species))), dtype=float)
        if F_bar.shape != (n_species, n_species):
            F_bar = np.zeros((n_species, n_species), dtype=float)

        nu, rop, wdot = _make_synthetic_dynamics(n_species, n_reactions, seed)
        i_reaction = reaction_importance(rop)
        dt = np.ones((rop.shape[0],), dtype=float)
        time = np.cumsum(dt)
        X = _reconstruct_state_from_wdot(wdot, dt)
        case_slices = [{"case_id": "synthetic", "start": 0, "end": int(X.shape[0]), "n_steps": int(X.shape[0])}]
        state_source = "synthetic"
        state_degraded = True
        state_fallback_reason = "synthetic_state_reconstruction"
        phase_context = {
            "mode": "synthetic",
            "applied": False,
            "phase_names": [],
            "weights": [],
            "n_phase": 0,
            "selected_phase": None,
            "used_i_reaction_by_phase": False,
            "source": "synthetic",
        }
    else:
        conditions = trace_ctx["conditions"]
        baseline_rows = trace_ctx["baseline_rows"]
        species_meta = trace_ctx["species_meta"]
        nu = np.asarray(trace_ctx["nu"], dtype=float)
        rop = np.asarray(trace_ctx["rop"], dtype=float)
        wdot = np.asarray(trace_ctx["wdot"], dtype=float)
        dt = np.asarray(trace_ctx["dt"], dtype=float)
        time = np.asarray(trace_ctx["time"], dtype=float)
        X = np.asarray(trace_ctx["X"], dtype=float)
        case_slices = list(trace_ctx.get("case_slices") or [])
        F_bar = np.asarray(trace_ctx["F_bar"], dtype=float)
        i_reaction = np.asarray(trace_ctx["i_reaction"], dtype=float)
        state_source = "trace_h5"
        state_degraded = False
        state_fallback_reason = None
        n_species = nu.shape[0]
        n_reactions = nu.shape[1]
        seed = int((cfg.get("search") or {}).get("seed", 7))
        phase_context = {
            "mode": "trace",
            "applied": False,
            "phase_names": [],
            "weights": [],
            "n_phase": 0,
            "selected_phase": None,
            "used_i_reaction_by_phase": False,
            "source": "trace_h5",
        }

    if np.asarray(i_reaction).shape != (n_reactions,):
        i_reaction = reaction_importance(rop)
    if dt.shape != (rop.shape[0],):
        dt = np.ones((rop.shape[0],), dtype=float)
        state_degraded = True
        state_fallback_reason = "invalid_dt_shape"
    if X.shape != (rop.shape[0], n_species):
        X = _reconstruct_state_from_wdot(wdot, dt)
        state_degraded = True
        state_fallback_reason = "invalid_state_shape"

    reduction_cfg = dict(cfg.get("reduction") or {})
    reduction_mode = str(reduction_cfg.get("mode", "baseline"))
    if reduction_mode not in {"baseline", "learnckpp", "pooling"}:
        raise ValueError("reduction.mode must be 'baseline' or 'learnckpp' or 'pooling'")
    learnckpp_cfg = dict(cfg.get("learnckpp") or {})
    pooling_cfg = dict(cfg.get("pooling") or {})
    physics_constraints_cfg = dict(cfg.get("physics_constraints") or {})
    if physics_constraints_cfg:
        pooling_constraint_cfg = dict(pooling_cfg.get("constraint_cfg") or {})
        pooling_constraint_hard = dict(pooling_constraint_cfg.get("hard") or {})
        if "phase_mixing_forbidden" in physics_constraints_cfg:
            pooling_constraint_hard["phase_mixing_forbidden"] = bool(physics_constraints_cfg.get("phase_mixing_forbidden"))
        if "surface_site_family_strict" in physics_constraints_cfg:
            pooling_constraint_hard["surface_site_family_strict"] = bool(physics_constraints_cfg.get("surface_site_family_strict"))
        pooling_constraint_cfg["hard"] = pooling_constraint_hard
        pooling_cfg["constraint_cfg"] = pooling_constraint_cfg

    surrogate_cfg = dict(cfg.get("surrogate") or {})
    surrogate_model_name = str(surrogate_cfg.get("model", "linear_ridge"))
    surrogate_l2 = float(surrogate_cfg.get("l2", 1.0e-6))
    surrogate_gain = float(surrogate_cfg.get("perturb_gain", 1.0))
    surrogate_blend = float(surrogate_cfg.get("blend_reference", 0.0))
    split_plan = _resolve_surrogate_split(conditions, eval_cfg)
    trained_surrogate: dict[str, Any] | None = None
    if surrogate_model_name == "linear_ridge" and str(split_plan.get("mode")) == "in_sample":
        trained_surrogate = fit_lightweight_surrogate(
            conditions,
            baseline_rows,
            qoi_cfg,
            l2=surrogate_l2,
            split_cfg={"mode": "in_sample"},
        )

    policy = _merge_policy_from_config(cfg)
    A = _build_element_matrix(species_meta)

    gate_cfg = dict(cfg.get("gate") or {})
    min_pass_rate = float(gate_cfg.get("min_pass_rate", 0.75))
    max_mean_rel = float(gate_cfg.get("max_mean_rel_diff", 0.40))
    max_cons = float(gate_cfg.get("max_conservation_violation", 0.0))
    physical_cfg = dict(eval_cfg.get("physical_gate") or {})
    physical_enabled = bool(physical_cfg.get("enabled", False))
    physical_max_cons = float(physical_cfg.get("max_conservation_violation", max_cons))
    physical_max_negative_steps = int(physical_cfg.get("max_negative_steps", 0))
    trace_meta = {
        "source": data_source,
        "trace_h5": str(cfg.get("trace_h5", "")),
        "species_names": [str(m.get("name", "")) for m in species_meta],
    }
    physics_profile = _resolve_physics_profile(cfg, trace_meta=trace_meta, mech_path=str(cfg.get("mechanism", "")))
    floors = _resolve_reduction_floors(physics_profile, n_species, n_reactions)
    balance_bands_base = _resolve_balance_bands(physics_profile, n_species, n_reactions, cfg)
    balance_bands = _resolve_dynamic_balance_bands(
        str(physics_profile.get("name", "default")),
        balance_bands_base,
        n_species,
        n_reactions,
        cfg,
    )
    essential_species = _resolve_essential_species(qoi_cfg, species_meta, cfg)
    gas_reaction_mask, surface_reaction_mask = _reaction_domain_masks(nu, species_meta)
    reaction_domain_counts_before = {
        "gas": int(np.sum(gas_reaction_mask)),
        "surface": int(np.sum(surface_reaction_mask)),
    }
    species_domain_counts_before = {
        "gas": int(np.sum([1 for m in species_meta if _is_gas_phase_name(m.get("phase"))])),
        "surface": int(np.sum([1 for m in species_meta if not _is_gas_phase_name(m.get("phase"))])),
    }

    stages = list((cfg.get("search") or {}).get("stages") or _stage_defaults())
    stage_rows: list[dict[str, Any]] = []
    gate_evidence_by_stage: dict[str, Any] = {}
    cluster_preview_by_stage: dict[str, list[dict[str, Any]]] = {}
    learnckpp_fallback_reasons: list[dict[str, Any]] = []
    learnckpp_fallback_enabled = bool(learnckpp_cfg.get("fallback_to_baseline_on_error", True))
    pooling_mapping_fallback_reasons: list[dict[str, Any]] = []
    pooling_mapping_fallback_enabled = bool(pooling_cfg.get("fallback_to_rule_based_on_error", True))
    learnckpp_adaptive_cfg = dict(learnckpp_cfg.get("adaptive_keep_ratio") or {})
    prev_stage_mean_rel: float | None = None
    prev_stage_physical: dict[str, Any] | None = None
    pooling_stage_artifacts: dict[str, str] = {}
    pooling_stage_metrics: dict[str, dict[str, Any]] = {}
    pooling_artifact_root = Path(str(pooling_cfg.get("artifact_dir", "artifacts/pooling"))) / args.run_id

    def _run_baseline_stage(
        *,
        stage_idx: int,
        stage_metric_drift: float,
        mapping: Any,
        prune_lambda: float,
        prune_threshold: float,
        prune_keep_ratio: float,
        prune_exact: bool,
    ) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]], Any, dict[str, Any], int, int, float, list[dict[str, Any]]]:
        keep_local, prune_local = train_prune_gate(
            nu,
            rop,
            wdot,
            lambda_l0=prune_lambda,
            threshold=prune_threshold,
            target_keep_ratio=prune_keep_ratio,
            enforce_target_exact=prune_exact,
            init_importance=i_reaction,
            seed=seed,
            return_details=True,
        )
        surrogate_rows_local, eval_summary_local, fold_metrics_local = _evaluate_surrogate_stage(
            surrogate_model_name=surrogate_model_name,
            conditions=conditions,
            baseline_rows=baseline_rows,
            qoi_cfg=qoi_cfg,
            eval_cfg=eval_cfg,
            split_plan=split_plan,
            surrogate_l2=surrogate_l2,
            surrogate_blend=surrogate_blend,
            metric_drift=stage_metric_drift,
            surrogate_gain=surrogate_gain,
            stage_idx=stage_idx,
            seed=seed,
            trained_surrogate=trained_surrogate,
        )
        physical_local = _evaluate_physical_gate(
            enabled=physical_enabled,
            nu=nu,
            rop=rop,
            dt=dt,
            A=A,
            X=X,
            S=np.asarray(mapping.S, dtype=float),
            keep_reactions=np.asarray(keep_local, dtype=bool),
            max_conservation_violation=physical_max_cons,
            max_negative_steps=physical_max_negative_steps,
            state_source=state_source,
            degraded=state_degraded,
            fallback_reason=state_fallback_reason,
        )
        overall_candidates_local = int(n_reactions)
        overall_selected_local = int(np.sum(keep_local))
        overall_select_ratio_local = float(overall_selected_local / max(overall_candidates_local, 1))
        return (
            np.asarray(keep_local, dtype=bool),
            dict(prune_local),
            list(surrogate_rows_local),
            eval_summary_local,
            dict(physical_local),
            overall_candidates_local,
            overall_selected_local,
            overall_select_ratio_local,
            list(fold_metrics_local),
        )

    for stage_idx, stage in enumerate(stages):
        name = str(stage.get("name", "unknown"))
        target_ratio = float(stage.get("target_ratio", 1.0))
        penalty_scale = float(stage.get("penalty_scale", 1.0))
        prune_lambda = float(stage.get("prune_lambda", 1.0e-3))
        prune_keep_ratio = float(stage.get("prune_keep_ratio", 1.0))
        prune_threshold = float(stage.get("prune_threshold", 0.5))
        prune_exact = bool(stage.get("prune_exact_keep", True))
        metric_drift = float(stage.get("metric_drift", 1.0))
        stage_floor_min_species = int(floors.get("min_species_after", 0))
        stage_floor_min_reactions = int(floors.get("min_reactions_after", 0))
        select_cfg_base = dict(learnckpp_cfg.get("select") or {})
        candidate_cfg_base = dict(learnckpp_cfg.get("candidate") or {})
        stage_min_keep_count_raw = _resolve_stage_scheduled_value(
            select_cfg_base.get("min_keep_count_by_stage"),
            stage_idx=stage_idx,
            stage_name=name,
            default=select_cfg_base.get("min_keep_count", 1),
        )
        stage_min_keep_count = max(int(stage_min_keep_count_raw), stage_floor_min_reactions)
        stage_min_candidates_floor_raw = _resolve_stage_scheduled_value(
            candidate_cfg_base.get("min_candidates_floor_by_stage"),
            stage_idx=stage_idx,
            stage_name=name,
            default=candidate_cfg_base.get("min_candidates_floor", 0),
        )
        stage_min_candidates_floor = max(int(stage_min_candidates_floor_raw), stage_floor_min_reactions)
        stage_min_flux_quantile_raw = _resolve_stage_scheduled_value(
            candidate_cfg_base.get("min_flux_quantile_by_stage"),
            stage_idx=stage_idx,
            stage_name=name,
            default=candidate_cfg_base.get("min_flux_quantile", 0.70),
        )
        stage_min_flux_quantile = float(np.clip(float(stage_min_flux_quantile_raw), 0.0, 1.0))
        stage_candidate_min_active_clusters_raw = _resolve_stage_scheduled_value(
            candidate_cfg_base.get("min_active_clusters_by_stage", candidate_cfg_base.get("min_active_clusters", 0)),
            stage_idx=stage_idx,
            stage_name=name,
            default=candidate_cfg_base.get("min_active_clusters", 0),
        )
        stage_candidate_min_active_clusters = max(int(stage_candidate_min_active_clusters_raw), 0)
        stage_min_active_clusters_raw = _resolve_stage_scheduled_value(
            select_cfg_base.get("min_active_clusters_by_stage", select_cfg_base.get("min_active_clusters", 0)),
            stage_idx=stage_idx,
            stage_name=name,
            default=select_cfg_base.get("min_active_clusters", 0),
        )
        stage_min_active_clusters = max(int(stage_min_active_clusters_raw), 0)
        stage_coverage_swap_steps_raw = _resolve_stage_scheduled_value(
            select_cfg_base.get("coverage_aware_swap_steps_by_stage", select_cfg_base.get("coverage_aware_swap_steps", 8)),
            stage_idx=stage_idx,
            stage_name=name,
            default=select_cfg_base.get("coverage_aware_swap_steps", 8),
        )
        stage_coverage_swap_steps = max(int(stage_coverage_swap_steps_raw), 0)
        stage_pool_train_base = dict(pooling_cfg.get("train") or {})
        stage_min_clusters_raw = _resolve_stage_scheduled_value(
            stage_pool_train_base.get("min_clusters_by_stage"),
            stage_idx=stage_idx,
            stage_name=name,
            default=stage_pool_train_base.get("min_clusters", 1),
        )
        stage_min_clusters = max(int(stage_min_clusters_raw), stage_floor_min_species)
        stage_coverage_target = float(
            _resolve_stage_scheduled_value(
                stage_pool_train_base.get("coverage_target_by_stage"),
                stage_idx=stage_idx,
                stage_name=name,
                default=stage_pool_train_base.get("coverage_target", 0.0),
            )
        )
        stage_coverage_max_clusters = int(
            _resolve_stage_scheduled_value(
                stage_pool_train_base.get("coverage_max_clusters_by_stage"),
                stage_idx=stage_idx,
                stage_name=name,
                default=stage_pool_train_base.get("coverage_max_clusters", n_species),
            )
        )
        stage_pool_min_active_clusters_raw = _resolve_stage_scheduled_value(
            stage_pool_train_base.get("min_active_clusters_by_stage", stage_pool_train_base.get("min_active_clusters", stage_min_clusters)),
            stage_idx=stage_idx,
            stage_name=name,
            default=stage_pool_train_base.get("min_active_clusters", stage_min_clusters),
        )
        stage_pool_min_active_clusters = max(int(stage_pool_min_active_clusters_raw), stage_min_clusters)
        stage_max_cluster_size_ratio_raw = _resolve_stage_scheduled_value(
            stage_pool_train_base.get("max_cluster_size_ratio_by_stage", stage_pool_train_base.get("max_cluster_size_ratio", 1.0)),
            stage_idx=stage_idx,
            stage_name=name,
            default=stage_pool_train_base.get("max_cluster_size_ratio", 1.0),
        )
        stage_max_cluster_size_ratio = float(np.clip(float(stage_max_cluster_size_ratio_raw), 0.0, 1.0))

        stage_pooling_metrics: dict[str, Any] = {}
        stage_pooling_artifact_path: str | None = None
        if reduction_mode == "pooling":
            stage_pooling_cfg = dict(pooling_cfg)
            stage_pooling_train = dict(stage_pooling_cfg.get("train") or {})
            stage_pooling_train["min_clusters"] = int(stage_min_clusters)
            stage_pooling_train["min_active_clusters"] = int(stage_pool_min_active_clusters)
            stage_pooling_train["coverage_target"] = float(max(0.0, min(1.0, stage_coverage_target)))
            stage_pooling_train["coverage_max_clusters"] = int(max(1, min(stage_coverage_max_clusters, n_species)))
            stage_pooling_train["max_cluster_size_ratio"] = float(stage_max_cluster_size_ratio)
            stage_pooling_cfg["train"] = stage_pooling_train
            artifact_path = pooling_artifact_root / f"{name}.npz"
            try:
                mapping, stage_pooling_metrics, saved_path = _fit_pooling_mapping(
                    nu=nu,
                    rop=rop,
                    F_bar=F_bar,
                    X=X,
                    wdot=wdot,
                    species_meta=species_meta,
                    target_ratio=target_ratio,
                    pooling_cfg=stage_pooling_cfg,
                    artifact_path=artifact_path,
                )
                if saved_path is not None:
                    stage_pooling_artifact_path = str(saved_path)
                    pooling_stage_artifacts[name] = str(saved_path)
            except Exception as exc:
                if not pooling_mapping_fallback_enabled:
                    raise
                mapping = fit_merge_mapping(
                    species_meta,
                    F_bar,
                    target_ratio=target_ratio,
                    policy=policy,
                    penalty_scale=penalty_scale,
                )
                stage_pooling_metrics = {
                    "status": "pooling_failed_rule_based_fallback",
                    "error": f"{type(exc).__name__}: {exc}",
                    "hard_ban_violations": int(mapping.meta.get("hard_ban_violations", 0)),
                    "train_metrics": {},
                    "model_info": {},
                    "graph_kind": str(pooling_cfg.get("graph", "species")),
                }
                pooling_mapping_fallback_reasons.append({"stage": name, "reason": stage_pooling_metrics["error"]})
        else:
            mapping = fit_merge_mapping(
                species_meta,
                F_bar,
                target_ratio=target_ratio,
                policy=policy,
                penalty_scale=penalty_scale,
            )

        keep: np.ndarray
        prune_details: dict[str, Any]
        physical_result: dict[str, Any]
        surrogate_rows: list[dict[str, Any]]
        eval_summary: Any
        overall_candidates = 0
        overall_selected = 0
        overall_select_ratio = 0.0
        stage_split_fold_metrics: list[dict[str, Any]] = []
        stage_metric_drift = metric_drift
        stage_reduction_mode = reduction_mode
        learnckpp_fallback_reason: str | None = None
        stage_pooling_fallback_reason: str | None = None
        learnckpp_target_keep_ratio = prune_keep_ratio
        learnckpp_keep_ratio_policy: dict[str, Any] = {}
        nu_balance = np.zeros((int(mapping.S.shape[1]), 0), dtype=float)

        if reduction_mode in {"learnckpp", "pooling"}:
            try:
                ydot_target = np.asarray(wdot @ np.asarray(mapping.S, dtype=float), dtype=float)
                learnckpp_features = _build_learnckpp_features(time=time, case_slices=case_slices, conditions=conditions)

                candidate_policy = {
                    "hard": dict((policy.get("hard") or {})),
                    "weights": dict((policy.get("weights") or {})),
                    "overlap_method": str(policy.get("overlap_method", "jaccard")),
                    "candidate": {
                        **candidate_cfg_base,
                        "min_candidates_floor": int(stage_min_candidates_floor),
                        "min_flux_quantile": float(stage_min_flux_quantile),
                        "min_active_clusters": int(stage_candidate_min_active_clusters),
                    },
                }
                cand = generate_overall_candidates(
                    nu=nu,
                    F_bar=F_bar,
                    S=np.asarray(mapping.S, dtype=float),
                    species_meta=species_meta,
                    policy=candidate_policy,
                )
                nu_overall = np.asarray(cand.get("nu_overall_candidates"), dtype=float)
                overall_candidates = int(nu_overall.shape[1])

                select_cfg = dict(select_cfg_base)
                if "fallback" not in select_cfg:
                    select_cfg["fallback"] = dict(learnckpp_cfg.get("fallback") or {})
                base_keep_ratio = float(select_cfg.get("target_keep_ratio", 0.30))
                staged_keep_ratio, keep_policy = _resolve_learnckpp_target_keep_ratio(
                    base_keep_ratio=base_keep_ratio,
                    prune_keep_ratio=prune_keep_ratio,
                    stage_idx=stage_idx,
                    data_source=data_source,
                    split_mode=str(split_plan.get("mode", "in_sample")),
                    max_mean_rel=max_mean_rel,
                    prev_mean_rel_diff=prev_stage_mean_rel,
                    prev_stage_physical=prev_stage_physical,
                    adaptive_cfg=learnckpp_adaptive_cfg,
                )
                if overall_candidates > 0:
                    min_ratio = 1.0 / float(overall_candidates)
                    staged_keep_ratio = float(max(staged_keep_ratio, min_ratio))
                    auto_tune_cfg = dict(learnckpp_adaptive_cfg.get("auto_tune") or {})
                    tuned_keep_ratio, auto_meta = _auto_tune_learnckpp_keep_ratio(
                        base_keep_ratio=staged_keep_ratio,
                        overall_candidates=overall_candidates,
                        min_keep_ratio=min_ratio,
                        max_keep_ratio=1.0,
                        data_source=data_source,
                        split_mode=str(split_plan.get("mode", "in_sample")),
                        prev_stage_physical=prev_stage_physical,
                        cfg=auto_tune_cfg,
                    )
                    staged_keep_ratio = float(max(tuned_keep_ratio, min_ratio))
                    keep_policy["auto_tune"] = auto_meta
                select_cfg["target_keep_ratio"] = staged_keep_ratio
                select_cfg["min_keep_count"] = int(stage_min_keep_count)
                select_cfg["min_active_clusters"] = int(stage_min_active_clusters)
                select_cfg["coverage_aware_swap_steps"] = int(stage_coverage_swap_steps)
                learnckpp_target_keep_ratio = staged_keep_ratio
                learnckpp_keep_ratio_policy = dict(keep_policy)
                select_cfg.setdefault("method", str(select_cfg.get("method", "hard_concrete")))
                select_cfg.setdefault("lambda_l0", float(select_cfg.get("lambda_l0", prune_lambda)))
                select_cfg.setdefault("seed", seed + stage_idx)
                s_arr = np.asarray(mapping.S, dtype=float)
                x_reduced_for_select = np.asarray(X @ s_arr, dtype=float)
                wdot_reduced_for_select = np.asarray(wdot @ s_arr, dtype=float)
                f_reduced_for_select = np.asarray(s_arr.T @ np.asarray(F_bar, dtype=float) @ s_arr, dtype=float)
                cluster_activity_weights = _build_species_activity_weights(
                    x_reduced_for_select,
                    wdot_reduced_for_select,
                    f_reduced_for_select,
                    cfg,
                )
                essential_cluster_mask_for_select = _build_essential_cluster_mask(s_arr, species_meta, essential_species)
                coverage_post_cfg = dict(learnckpp_cfg.get("coverage_postselect") or {})
                stage_cov_max_keep = _resolve_stage_scheduled_value(
                    coverage_post_cfg.get("max_keep_count_by_stage"),
                    stage_idx=stage_idx,
                    stage_name=name,
                    default=coverage_post_cfg.get("max_keep_count", overall_candidates),
                )
                stage_cov_weighted_target = _resolve_stage_scheduled_value(
                    coverage_post_cfg.get("target_weighted_coverage_by_stage"),
                    stage_idx=stage_idx,
                    stage_name=name,
                    default=coverage_post_cfg.get(
                        "target_weighted_coverage",
                        balance_bands.get("min_weighted_active_species_coverage", 0.0),
                    ),
                )
                stage_cov_essential_target = _resolve_stage_scheduled_value(
                    coverage_post_cfg.get("target_essential_coverage_by_stage"),
                    stage_idx=stage_idx,
                    stage_name=name,
                    default=coverage_post_cfg.get(
                        "target_essential_coverage",
                        balance_bands.get("min_essential_species_coverage", 0.0),
                    ),
                )
                select_cfg["coverage_postselect"] = {
                    "enabled": bool(coverage_post_cfg.get("enabled", True)),
                    "mode": str(balance_bands.get("balance_mode", "binary")),
                    "target_weighted_coverage": float(stage_cov_weighted_target),
                    "target_essential_coverage": float(stage_cov_essential_target),
                    "cluster_weights": np.asarray(cluster_activity_weights, dtype=float).tolist(),
                    "essential_cluster_mask": np.asarray(essential_cluster_mask_for_select, dtype=bool).tolist(),
                    "max_keep_count": int(max(1, min(int(stage_cov_max_keep), overall_candidates))),
                }

                keep, select_details = select_sparse_overall(
                    nu_cand=nu_overall,
                    ydot_target=ydot_target,
                    features=learnckpp_features,
                    cfg=select_cfg,
                )

                if nu_overall.shape[1] == 0:
                    nu_sel = np.zeros((int(mapping.S.shape[1]), 1), dtype=float)
                    rates_pred = np.zeros((ydot_target.shape[0], 1), dtype=float)
                    y_pred = np.maximum(0.0, np.asarray(X, dtype=float) @ np.asarray(mapping.S, dtype=float))
                    select_status = "empty_candidates"
                    keep = np.zeros((0,), dtype=bool)
                    nu_balance = np.zeros((int(mapping.S.shape[1]), 0), dtype=float)
                else:
                    rates_target = np.asarray(select_details.get("rates_target"), dtype=float)
                    if rates_target.shape != (ydot_target.shape[0], nu_overall.shape[1]):
                        rates_target = np.asarray(ydot_target @ np.linalg.pinv(nu_overall.T), dtype=float)

                    if keep.shape != (nu_overall.shape[1],):
                        keep = np.zeros((nu_overall.shape[1],), dtype=bool)
                    if not np.any(keep):
                        keep[0] = True

                    nu_sel = np.asarray(nu_overall[:, keep], dtype=float)
                    nu_balance = np.asarray(nu_sel, dtype=float)
                    rates_target_sel = np.asarray(rates_target[:, keep], dtype=float)
                    rate_cfg = dict(learnckpp_cfg.get("rate") or {})
                    model_artifact = fit_rate_model(learnckpp_features, rates_target_sel, rate_cfg)
                    model_artifact["sim_features"] = learnckpp_features.tolist()

                    proj_cfg = dict(learnckpp_cfg.get("projection") or {})
                    proj_cfg.setdefault("enabled", True)
                    proj_cfg.setdefault("max_iter", 4)
                    proj_cfg.setdefault("clip_nonnegative", True)
                    proj_cfg["A"] = np.asarray(A @ np.asarray(mapping.S, dtype=float), dtype=float)
                    proj_cfg["reference"] = np.asarray(X[0] @ np.asarray(mapping.S, dtype=float), dtype=float)

                    y_pred, _ = simulate_reduced(
                        y0=np.asarray(X[0] @ np.asarray(mapping.S, dtype=float), dtype=float),
                        time=time,
                        model_artifact=model_artifact,
                        nu_overall_sel=nu_sel,
                        proj_cfg=proj_cfg,
                    )
                    rates_pred = predict_rates(model_artifact, learnckpp_features)
                    if rates_pred.shape[1] != nu_sel.shape[1]:
                        rates_pred = np.asarray(rates_target_sel, dtype=float)
                    select_status = str(select_details.get("status"))

                    y_ref = np.asarray(X @ np.asarray(mapping.S, dtype=float), dtype=float)
                    rel = np.abs(y_pred - y_ref) / (np.abs(y_ref) + 1.0e-12)
                    stage_metric_drift = float(np.clip(1.0 + float(np.mean(rel)), 1.0, 1.30))

                overall_selected = int(np.sum(keep)) if keep.size else 0
                overall_select_ratio = float(overall_selected / max(overall_candidates, 1))
                prune_details = {
                    "status": select_status,
                    "keep_count": overall_selected,
                    "keep_ratio": overall_select_ratio,
                    "mode": "learnckpp",
                }

                surrogate_rows, eval_summary, stage_split_fold_metrics = _evaluate_surrogate_stage(
                    surrogate_model_name=surrogate_model_name,
                    conditions=conditions,
                    baseline_rows=baseline_rows,
                    qoi_cfg=qoi_cfg,
                    eval_cfg=eval_cfg,
                    split_plan=split_plan,
                    surrogate_l2=surrogate_l2,
                    surrogate_blend=surrogate_blend,
                    metric_drift=stage_metric_drift,
                    surrogate_gain=surrogate_gain,
                    stage_idx=stage_idx,
                    seed=seed,
                    trained_surrogate=trained_surrogate,
                )

                keep_gate = np.ones((int(nu_sel.shape[1]),), dtype=bool)
                physical_result = _evaluate_physical_gate(
                    enabled=physical_enabled,
                    nu=nu_sel,
                    rop=rates_pred,
                    dt=dt,
                    A=np.asarray(A @ np.asarray(mapping.S, dtype=float), dtype=float),
                    X=np.asarray(y_pred, dtype=float),
                    S=np.eye(int(mapping.S.shape[1]), dtype=float),
                    keep_reactions=keep_gate,
                    max_conservation_violation=physical_max_cons,
                    max_negative_steps=physical_max_negative_steps,
                    state_source=f"{state_source}:learnckpp",
                    degraded=state_degraded,
                    fallback_reason=state_fallback_reason,
                )
            except Exception as exc:
                if not learnckpp_fallback_enabled:
                    raise
                stage_reduction_mode = "baseline_fallback"
                learnckpp_fallback_reason = f"{type(exc).__name__}: {exc}"
                learnckpp_fallback_reasons.append({"stage": name, "reason": learnckpp_fallback_reason})
                if reduction_mode == "pooling":
                    stage_pooling_fallback_reason = learnckpp_fallback_reason
                (
                    keep,
                    prune_details,
                    surrogate_rows,
                    eval_summary,
                    physical_result,
                    overall_candidates,
                    overall_selected,
                    overall_select_ratio,
                    stage_split_fold_metrics,
                ) = _run_baseline_stage(
                    stage_idx=stage_idx,
                    stage_metric_drift=metric_drift,
                    mapping=mapping,
                    prune_lambda=prune_lambda,
                    prune_threshold=prune_threshold,
                    prune_keep_ratio=prune_keep_ratio,
                    prune_exact=prune_exact,
                )
                fallback_prefix = ("pooling" if reduction_mode == "pooling" else "learnckpp")
                prune_details["status"] = f"{fallback_prefix}_failed_baseline:{prune_details.get('status')}"
                learnckpp_target_keep_ratio = prune_keep_ratio
                learnckpp_keep_ratio_policy = {"enabled": False, "fallback": True}
                nu_balance = np.asarray(np.asarray(mapping.S, dtype=float).T @ np.asarray(nu[:, keep], dtype=float), dtype=float)
        else:
            (
                keep,
                prune_details,
                surrogate_rows,
                eval_summary,
                physical_result,
                overall_candidates,
                overall_selected,
                overall_select_ratio,
                stage_split_fold_metrics,
            ) = _run_baseline_stage(
                stage_idx=stage_idx,
                stage_metric_drift=stage_metric_drift,
                mapping=mapping,
                prune_lambda=prune_lambda,
                prune_threshold=prune_threshold,
                prune_keep_ratio=prune_keep_ratio,
                prune_exact=prune_exact,
            )
            nu_balance = np.asarray(np.asarray(mapping.S, dtype=float).T @ np.asarray(nu[:, keep], dtype=float), dtype=float)

        cons_violation = float(physical_result["conservation_violation"])
        negative_steps = int(physical_result["negative_steps"])
        species_after = int(mapping.S.shape[1])
        reactions_after = int(overall_selected)
        keep_arr = np.asarray(keep, dtype=bool).reshape(-1)
        reaction_domain_split_available = bool(keep_arr.shape == gas_reaction_mask.shape)
        if reaction_domain_split_available:
            gas_reactions_after = int(np.sum(keep_arr & gas_reaction_mask))
            surface_reactions_after = int(np.sum(keep_arr & surface_reaction_mask))
        else:
            gas_reactions_after = -1
            surface_reactions_after = -1
        cluster_domain_counts_after = _cluster_domain_counts(np.asarray(mapping.S, dtype=float), species_meta)
        gas_species_after = int(cluster_domain_counts_after.get("gas", 0))
        surface_species_after = int(cluster_domain_counts_after.get("surface", 0))
        floor_min_species = int(floors.get("min_species_after", 0))
        floor_min_reactions = int(floors.get("min_reactions_after", 0))
        floor_violations: list[str] = []
        if species_after < floor_min_species:
            floor_violations.append("min_species_after")
        if reactions_after < floor_min_reactions:
            floor_violations.append("min_reactions_after")
        floor_passed = len(floor_violations) == 0
        balance_metrics = _compute_structural_balance_metrics(
            nu_reduced=nu_balance,
            species_after=species_after,
            reactions_after=reactions_after,
            activity_weights=_build_species_activity_weights(
                np.asarray(X @ np.asarray(mapping.S, dtype=float), dtype=float),
                np.asarray(wdot @ np.asarray(mapping.S, dtype=float), dtype=float),
                np.asarray(np.asarray(mapping.S, dtype=float).T @ np.asarray(F_bar, dtype=float) @ np.asarray(mapping.S, dtype=float), dtype=float),
                cfg,
            ),
            essential_cluster_mask=_build_essential_cluster_mask(np.asarray(mapping.S, dtype=float), species_meta, essential_species),
            S_reduced=np.asarray(mapping.S, dtype=float),
            species_before=int(n_species),
            top_weight_mass_ratio=float(balance_bands.get("top_weight_mass_ratio", 0.80)),
            balance_mode=str(balance_bands.get("balance_mode", "binary")),
        )
        balance_result = _evaluate_balance_gate(balance_metrics, balance_bands)
        balance_passed = bool(balance_result.get("passed", True))
        balance_violations = list(balance_result.get("violations") or [])
        balance_mode = str(balance_result.get("balance_mode", balance_bands.get("balance_mode", "binary")))
        balance_band_effective = dict(balance_result.get("bands") or {})
        min_ratio = float(balance_band_effective.get("min_reaction_species_ratio", 0.0))
        max_ratio = float(balance_band_effective.get("max_reaction_species_ratio", 1.0e9))
        cov_floor = float(balance_band_effective.get("min_active_species_coverage", 0.0))
        weighted_cov_floor = float(balance_band_effective.get("min_weighted_active_species_coverage", 0.0))
        top_weighted_cov_floor = float(balance_band_effective.get("min_active_species_coverage_top_weighted", 0.0))
        essential_cov_floor = float(balance_band_effective.get("min_essential_species_coverage_effective", balance_band_effective.get("min_essential_species_coverage", 0.0)))
        max_cluster_ratio_floor = float(balance_band_effective.get("max_cluster_size_ratio", 1.0e9))
        rank_floor = float(balance_band_effective.get("min_nu_rank_ratio", 0.0))
        ratio_val = float(balance_metrics.get("reaction_species_ratio", 0.0))
        cov_val = float(balance_metrics.get("active_species_coverage", 0.0))
        weighted_cov_val = float(balance_metrics.get("weighted_active_species_coverage", cov_val))
        top_weighted_cov_val = float(balance_metrics.get("active_species_coverage_top_weighted", weighted_cov_val))
        essential_cov_val = float(balance_metrics.get("essential_species_coverage", 1.0))
        rank_val = float(balance_metrics.get("nu_rank_ratio", 0.0))
        max_cluster_ratio_val = float(balance_metrics.get("max_cluster_size_ratio", 0.0))
        margin_vector = _compute_balance_margin_vector(balance_metrics, balance_band_effective)
        balance_margin = float(margin_vector.get("balance_margin", 0.0))
        coverage_margin = float(margin_vector.get("coverage_margin", 0.0))
        cluster_size_margin = float(margin_vector.get("cluster_size_margin", 0.0))
        balance_margin_detail = dict(margin_vector.get("detail") or {})
        cluster_guard_violations = [v for v in balance_violations if v == "max_cluster_size_ratio"]
        cluster_guard_passed = len(cluster_guard_violations) == 0

        gate_evidence_by_stage[name] = {
            **physical_result,
            "split_mode": str(split_plan.get("mode")),
            "split_fallback_reason": split_plan.get("fallback_reason"),
            "split_effective_kfolds": int(split_plan.get("effective_kfolds", 0)),
            "split_fold_sizes": list(split_plan.get("fold_sizes") or []),
            "kfold_fold_metrics": list(stage_split_fold_metrics),
            "surrogate_rows": int(len(surrogate_rows)),
            "reduction_mode": stage_reduction_mode,
            "learnckpp_overall_candidates": int(overall_candidates),
            "learnckpp_overall_selected": int(overall_selected),
            "learnckpp_overall_select_ratio": float(overall_select_ratio),
            "reaction_domain_counts_before": reaction_domain_counts_before,
            "reaction_domain_counts_after": {"gas": gas_reactions_after, "surface": surface_reactions_after},
            "reaction_domain_split_available": bool(reaction_domain_split_available),
            "species_domain_counts_before": species_domain_counts_before,
            "species_domain_counts_after": {"gas": gas_species_after, "surface": surface_species_after},
            "learnckpp_select_status": str(prune_details.get("status")),
            "learnckpp_fallback_reason": learnckpp_fallback_reason,
            "learnckpp_target_keep_ratio": float(learnckpp_target_keep_ratio),
            "learnckpp_keep_ratio_policy": learnckpp_keep_ratio_policy,
            "pooling_hard_ban_violations": int(stage_pooling_metrics.get("hard_ban_violations", 0)),
            "pooling_constraint_loss": float((stage_pooling_metrics.get("train_metrics") or {}).get("constraint_loss", 0.0)),
            "pooling_clusters": int((stage_pooling_metrics.get("train_metrics") or {}).get("n_clusters", mapping.S.shape[1])),
            "pooling_graph_kind": str(stage_pooling_metrics.get("graph_kind", "")),
            "pooling_model_type": str((stage_pooling_metrics.get("model_info") or {}).get("model_type", "")),
            "pooling_fallback_reason": stage_pooling_fallback_reason,
            "pooling_artifact_path": stage_pooling_artifact_path,
            "floor_passed": bool(floor_passed),
            "floor_min_species": floor_min_species,
            "floor_min_reactions": floor_min_reactions,
            "floor_violations": floor_violations,
            "balance_gate_enabled": bool(balance_result.get("enabled", False)),
            "balance_gate_passed": bool(balance_passed),
            "balance_violations": balance_violations,
            "cluster_guard_violations": cluster_guard_violations,
            "cluster_guard_passed": bool(cluster_guard_passed),
            "balance_metrics": balance_metrics,
            "balance_bands": balance_band_effective,
            "balance_margin_detail": balance_margin_detail,
        }
        if reduction_mode == "pooling":
            pooling_stage_metrics[name] = dict(stage_pooling_metrics)

        hard_ban_violations = int(mapping.meta.get("hard_ban_violations", 0))
        cluster_preview_by_stage[name] = _cluster_preview(
            S=np.asarray(mapping.S, dtype=float),
            pool_meta=list(mapping.pool_meta or []),
            species_meta=species_meta,
            max_clusters=int((cfg.get("reporting") or {}).get("max_cluster_preview", 8)),
            max_members=int((cfg.get("reporting") or {}).get("max_members_preview", 6)),
        )
        physical_pass = bool(physical_result["passed"])
        gate_passed = (
            eval_summary.pass_rate >= min_pass_rate
            and (eval_summary.mean_rel_diff or 0.0) <= max_mean_rel
            and hard_ban_violations == 0
            and (physical_pass if physical_enabled else True)
            and floor_passed
            and balance_passed
        )

        stage_rows.append(
            {
                "stage": name,
                "reduction_mode": stage_reduction_mode,
                "species_before": int(n_species),
                "species_after": species_after,
                "gas_species_before": int(species_domain_counts_before["gas"]),
                "surface_species_before": int(species_domain_counts_before["surface"]),
                "gas_species_after": gas_species_after,
                "surface_species_after": surface_species_after,
                "reactions_before": int(n_reactions),
                "reactions_after": reactions_after,
                "gas_reactions_before": int(reaction_domain_counts_before["gas"]),
                "surface_reactions_before": int(reaction_domain_counts_before["surface"]),
                "gas_reactions_after": gas_reactions_after,
                "surface_reactions_after": surface_reactions_after,
                "reaction_domain_split_available": bool(reaction_domain_split_available),
                "pass_rate": float(eval_summary.pass_rate),
                "mean_rel_diff": float(eval_summary.mean_rel_diff or 0.0),
                "max_rel_diff": float(eval_summary.max_rel_diff or 0.0),
                "conservation_violation": float(cons_violation),
                "negative_steps": negative_steps,
                "physical_gate_enabled": bool(physical_enabled),
                "physical_gate_passed": physical_pass,
                "physical_degraded": bool(physical_result["degraded"]),
                "hard_ban_violations": hard_ban_violations,
                "floor_passed": bool(floor_passed),
                "floor_min_species": floor_min_species,
                "floor_min_reactions": floor_min_reactions,
                "floor_violations": ";".join(floor_violations),
                "balance_gate_passed": bool(balance_passed),
                "balance_violations": ";".join(balance_violations),
                "reaction_species_ratio": float(ratio_val),
                "active_species_coverage": float(cov_val),
                "weighted_active_species_coverage": float(weighted_cov_val),
                "active_species_coverage_top_weighted": float(top_weighted_cov_val),
                "essential_species_coverage": float(essential_cov_val),
                "nu_rank_ratio": float(rank_val),
                "max_cluster_size_ratio": float(max_cluster_ratio_val),
                "balance_mode": balance_mode,
                "balance_margin": float(balance_margin),
                "coverage_margin": float(coverage_margin),
                "cluster_size_margin": float(cluster_size_margin),
                "cluster_guard_passed": bool(cluster_guard_passed),
                "cluster_guard_violations": ";".join(cluster_guard_violations),
                "balance_dynamic_applied": bool(balance_band_effective.get("balance_dynamic_applied", False)),
                "balance_dynamic_complexity": float(balance_band_effective.get("balance_dynamic_complexity", 0.0)),
                "rs_upper_effective": float(balance_band_effective.get("rs_upper_effective", max_ratio)),
                "active_cov_effective_floor": float(balance_band_effective.get("active_cov_effective_floor", cov_floor)),
                "split_mode": str(split_plan.get("mode")),
                "effective_kfolds": int(split_plan.get("effective_kfolds", 0)),
                "qoi_metrics_count": int(getattr(eval_summary, "qoi_metrics_count", 0)),
                "integral_qoi_count": int(qoi_integral_count),
                "gate_passed": bool(gate_passed),
                "target_ratio": target_ratio,
                "penalty_scale": penalty_scale,
                "prune_lambda": prune_lambda,
                "prune_keep_ratio": prune_keep_ratio,
                "prune_threshold": prune_threshold,
                "prune_exact_keep": prune_exact,
                "prune_status": str(prune_details.get("status")),
                "overall_candidates": int(overall_candidates),
                "overall_selected": int(overall_selected),
                "overall_select_ratio": float(overall_select_ratio),
                "metric_drift_effective": float(stage_metric_drift),
                "learnckpp_fallback_reason": learnckpp_fallback_reason,
                "learnckpp_target_keep_ratio": float(learnckpp_target_keep_ratio),
                "pooling_clusters": int((stage_pooling_metrics.get("train_metrics") or {}).get("n_clusters", mapping.S.shape[1])),
                "pooling_constraint_loss": float((stage_pooling_metrics.get("train_metrics") or {}).get("constraint_loss", 0.0)),
                "pooling_hard_ban_violations": int(stage_pooling_metrics.get("hard_ban_violations", 0)),
                "pooling_artifact_path": stage_pooling_artifact_path,
                "_floors": floors,
                "_selection_max_mean_rel": max_mean_rel,
            }
        )
        prev_stage_mean_rel = float(eval_summary.mean_rel_diff or 0.0)
        prev_stage_physical = dict(physical_result)

    selection_result = _select_stage_physics_first(stage_rows, cfg)
    selected = dict(selection_result["selected"])
    for key in ("_floors", "_selection_max_mean_rel"):
        selected.pop(key, None)
    for row in stage_rows:
        row.pop("_floors", None)
        row.pop("_selection_max_mean_rel", None)
    report_dir = Path(cfg.get("report_dir", "reports")) / args.run_id
    selected_stage_name = str(selected["stage"])
    selected_gate_evidence = dict(gate_evidence_by_stage.get(selected_stage_name) or {})
    pareto_candidates = [dict(r) for r in list(selection_result.get("pareto_candidates") or [])]
    for row in pareto_candidates:
        row.pop("_floors", None)
        row.pop("_selection_max_mean_rel", None)
    candidate_trend = [
        {
            "stage": str(row.get("stage")),
            "species_after": int(row.get("species_after", 0)),
            "gas_species_after": int(row.get("gas_species_after", 0)),
            "surface_species_after": int(row.get("surface_species_after", 0)),
            "reactions_after": int(row.get("reactions_after", 0)),
            "gas_reactions_after": int(row.get("gas_reactions_after", 0)),
            "surface_reactions_after": int(row.get("surface_reactions_after", 0)),
            "overall_candidates": int(row.get("overall_candidates", 0)),
            "overall_selected": int(row.get("overall_selected", 0)),
            "overall_select_ratio": float(row.get("overall_select_ratio", 0.0)),
            "mean_rel_diff": float(row.get("mean_rel_diff", 0.0)),
            "floor_passed": bool(row.get("floor_passed", True)),
            "balance_gate_passed": bool(row.get("balance_gate_passed", True)),
            "cluster_guard_passed": bool(row.get("cluster_guard_passed", True)),
            "weighted_active_species_coverage": float(row.get("weighted_active_species_coverage", row.get("active_species_coverage", 0.0))),
            "active_species_coverage_top_weighted": float(
                row.get("active_species_coverage_top_weighted", row.get("weighted_active_species_coverage", row.get("active_species_coverage", 0.0)))
            ),
            "essential_species_coverage": float(row.get("essential_species_coverage", 1.0)),
            "max_cluster_size_ratio": float(row.get("max_cluster_size_ratio", 0.0)),
            "balance_mode": str(row.get("balance_mode", balance_bands.get("balance_mode", "binary"))),
            "selection_score": float(row.get("selection_score", 0.0)),
            "balance_margin": float(row.get("balance_margin", 0.0)),
            "balance_dynamic_applied": bool(row.get("balance_dynamic_applied", False)),
            "balance_dynamic_complexity": float(row.get("balance_dynamic_complexity", 0.0)),
            "rs_upper_effective": float(row.get("rs_upper_effective", 0.0)),
            "active_cov_effective_floor": float(row.get("active_cov_effective_floor", 0.0)),
        }
        for row in stage_rows
    ]
    selected["balance_metrics"] = {
        "reaction_species_ratio": float(selected.get("reaction_species_ratio", 0.0)),
        "active_species_coverage": float(selected.get("active_species_coverage", 0.0)),
        "weighted_active_species_coverage": float(selected.get("weighted_active_species_coverage", selected.get("active_species_coverage", 0.0))),
        "active_species_coverage_top_weighted": float(
            selected.get("active_species_coverage_top_weighted", selected.get("weighted_active_species_coverage", selected.get("active_species_coverage", 0.0)))
        ),
        "essential_species_coverage": float(selected.get("essential_species_coverage", 1.0)),
        "nu_rank_ratio": float(selected.get("nu_rank_ratio", 0.0)),
            "max_cluster_size_ratio": float(selected.get("max_cluster_size_ratio", 0.0)),
            "balance_mode": str(selected.get("balance_mode", balance_bands.get("balance_mode", "binary"))),
            "balance_gate_passed": bool(selected.get("balance_gate_passed", True)),
            "balance_violations": str(selected.get("balance_violations", "")),
            "balance_dynamic_applied": bool(selected.get("balance_dynamic_applied", False)),
            "balance_dynamic_complexity": float(selected.get("balance_dynamic_complexity", 0.0)),
            "rs_upper_effective": float(selected.get("rs_upper_effective", 0.0)),
            "active_cov_effective_floor": float(selected.get("active_cov_effective_floor", 0.0)),
        }

    summary_payload = {
        "gate_passed": bool(selected["gate_passed"]),
        "hard_ban_violations": int(selected["hard_ban_violations"]),
        "reduction_mode": reduction_mode,
        "gate": {
            "min_pass_rate": min_pass_rate,
            "max_mean_rel_diff": max_mean_rel,
            "max_conservation_violation": (physical_max_cons if physical_enabled else max_cons),
            "max_negative_steps": (physical_max_negative_steps if physical_enabled else None),
            "physics_floors": floors,
            "balance_bands": balance_bands,
            "balance_bands_base": balance_bands_base,
            "selection_policy": str(selection_result.get("policy", "")),
            "selection_tie_breakers": list(selection_result.get("tie_breakers") or []),
            "physics_profile": physics_profile,
        },
        "selected_metrics": selected,
        "data_source": data_source,
        "surrogate_model": surrogate_model_name,
        "surrogate_split": split_plan,
        "qoi_integral_count": int(qoi_integral_count),
        "qoi_integral_keys": qoi_integral_keys,
        "phase_context": phase_context,
        "domain_counts": {
            "species_before": species_domain_counts_before,
            "reactions_before": reaction_domain_counts_before,
            "selected_species_after": {
                "gas": int(selected.get("gas_species_after", 0)),
                "surface": int(selected.get("surface_species_after", 0)),
            },
            "selected_reactions_after": {
                "gas": int(selected.get("gas_reactions_after", 0)),
                "surface": int(selected.get("surface_reactions_after", 0)),
            },
        },
        "gate_evidence": {
            "physical_gate_enabled": bool(physical_enabled),
            "selected_stage": selected_stage_name,
            "selected_stage_evidence": selected_gate_evidence,
            "state_source": state_source,
            "state_degraded": state_degraded,
            "state_fallback_reason": state_fallback_reason,
            "case_slices_count": int(len(case_slices)),
            "floor_violations": list(selected_gate_evidence.get("floor_violations") or []),
            "balance_violations": list(selected_gate_evidence.get("balance_violations") or []),
            "cluster_guard_violations": list(selected_gate_evidence.get("cluster_guard_violations") or []),
            "kfold_fold_metrics": {
                "selected_stage": selected_stage_name,
                "selected": list(selected_gate_evidence.get("kfold_fold_metrics") or []),
                "per_stage": {
                    str(stage): list((evidence or {}).get("kfold_fold_metrics") or [])
                    for stage, evidence in gate_evidence_by_stage.items()
                },
            },
            "balance_margin_detail": {
                "selected_stage": selected_stage_name,
                "selected": dict(selected_gate_evidence.get("balance_margin_detail") or {}),
                "per_stage": {
                    str(stage): dict((evidence or {}).get("balance_margin_detail") or {})
                    for stage, evidence in gate_evidence_by_stage.items()
                },
            },
            "pareto_candidates": pareto_candidates,
            "per_stage": gate_evidence_by_stage,
        },
        "reduction_trace": {
            "candidate_trend": candidate_trend,
            "cluster_preview": {
                "selected_stage": selected_stage_name,
                "selected_stage_clusters": list(cluster_preview_by_stage.get(selected_stage_name) or []),
                "per_stage": cluster_preview_by_stage,
            },
        },
        "failure_reason": (
            "pooling_mapping_failed_rule_based_fallback" if pooling_mapping_fallback_reasons
            else ("learnckpp_stage_failed_fallback_baseline" if learnckpp_fallback_reasons else None)
        ),
    }
    if reduction_mode == "learnckpp":
        summary_payload["learnckpp_fallback"] = {
            "enabled": bool(learnckpp_fallback_enabled),
            "triggered": bool(learnckpp_fallback_reasons),
            "reasons": learnckpp_fallback_reasons,
        }
    if reduction_mode == "learnckpp":
        summary_payload["learnckpp_metrics"] = {
            "overall_candidates": int(selected.get("overall_candidates", 0)),
            "overall_selected": int(selected.get("overall_selected", 0)),
            "overall_select_ratio": float(selected.get("overall_select_ratio", 0.0)),
            "select_status": str(selected.get("prune_status", "")),
            "target_keep_ratio": float(selected.get("learnckpp_target_keep_ratio", 0.0)),
            "adaptive_keep_ratio": {
                "enabled": bool(learnckpp_adaptive_cfg.get("enabled", True)),
                "config": learnckpp_adaptive_cfg,
            },
            "selected_stage": selected_stage_name,
        }
    if reduction_mode == "pooling":
        selected_pool_metrics = dict(pooling_stage_metrics.get(selected_stage_name) or {})
        summary_payload["pooling_metrics"] = {
            "overall_candidates": int(selected.get("overall_candidates", 0)),
            "overall_selected": int(selected.get("overall_selected", 0)),
            "overall_select_ratio": float(selected.get("overall_select_ratio", 0.0)),
            "n_clusters": int((selected_pool_metrics.get("train_metrics") or {}).get("n_clusters", selected.get("species_after", 0))),
            "constraint_loss": float((selected_pool_metrics.get("train_metrics") or {}).get("constraint_loss", 0.0)),
            "hard_ban_violations": int(selected.get("pooling_hard_ban_violations", 0)),
            "graph_kind": str(selected_pool_metrics.get("graph_kind", "")),
            "model_type": str((selected_pool_metrics.get("model_info") or {}).get("model_type", "")),
            "selected_stage": selected_stage_name,
            "mapping_fallback_enabled": bool(pooling_mapping_fallback_enabled),
            "mapping_fallback_triggered": bool(pooling_mapping_fallback_reasons),
            "mapping_fallback_reasons": pooling_mapping_fallback_reasons,
        }
        summary_payload["pooling_artifact_path"] = str(
            pooling_stage_artifacts.get(selected_stage_name, selected.get("pooling_artifact_path") or "")
        )

    write_report(
        report_dir,
        run_id=args.run_id,
        stage_rows=stage_rows,
        selected_stage=str(selected["stage"]),
        summary_payload=summary_payload,
    )

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
