from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time as pytime
from typing import Any

from datetime import datetime, timezone

import numpy as np
import yaml

from cantera_model.eval.cantera_runner import load_conditions
from cantera_model.eval.diagnostic_schema import validate_summary_schema
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
from cantera_model.reduction.pooling.train import _refine_cluster_balance_swap, train_pooling_assignment
from cantera_model.reduction.merge_free import DEFAULT_POLICY, fit_merge_mapping
from cantera_model.reduction.prune_gate import train_prune_gate
from cantera_model.types import ReductionMapping


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    return data


def _resolve_evaluation_contract(eval_cfg: dict[str, Any]) -> dict[str, Any]:
    contract = dict(eval_cfg.get("contract") or {})
    contract.setdefault("version", "v1")
    contract.setdefault("enforce", False)
    contract.setdefault("invariants_profile", "strict_physical_v1")
    contract.setdefault("evaluation_profile", "tiered_v35")
    contract.setdefault("run_policy_profile", "adaptive_kfold_strict_v1")
    contract.setdefault("diagnostic_schema_strict", False)
    eval_cfg["contract"] = contract
    return contract


def _validate_evaluation_contract(eval_cfg: dict[str, Any], contract: dict[str, Any]) -> None:
    if not bool(contract.get("enforce", False)):
        return
    required = (
        "version",
        "invariants_profile",
        "evaluation_profile",
        "run_policy_profile",
        "diagnostic_schema_strict",
    )
    missing = [key for key in required if key not in contract]
    if missing:
        raise ValueError(
            "evaluation.contract missing required keys when enforce=true: " + ", ".join(missing)
        )
    eval_profile = str(contract.get("evaluation_profile", "")).strip().lower()
    if eval_profile == "tiered_v35":
        err = dict(eval_cfg.get("error_aggregation") or {})
        if str(err.get("mode", "tiered")).strip().lower() != "tiered":
            raise ValueError("evaluation.contract requires error_aggregation.mode=tiered")
        if not bool(err.get("require_explicit_thresholds", False)):
            raise ValueError(
                "evaluation.contract requires error_aggregation.require_explicit_thresholds=true"
            )
    run_profile = str(contract.get("run_policy_profile", "")).strip().lower()
    if run_profile == "adaptive_kfold_strict_v1":
        split = dict(eval_cfg.get("surrogate_split") or {})
        if str(split.get("mode", "")).strip().lower() != "adaptive_kfold":
            raise ValueError("evaluation.contract requires surrogate_split.mode=adaptive_kfold")
        if not bool(split.get("enforce_explicit_policy", False)):
            raise ValueError(
                "evaluation.contract requires surrogate_split.enforce_explicit_policy=true"
            )


def _load_metric_taxonomy_profile(
    eval_cfg: dict[str, Any],
    *,
    config_parent: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    taxonomy_cfg = dict(eval_cfg.get("metric_taxonomy") or {})
    source = str(taxonomy_cfg.get("source", "legacy_builtin")).strip().lower()
    profile = str(taxonomy_cfg.get("profile", "legacy_builtin")).strip() or "legacy_builtin"
    if source != "shared_yaml":
        resolved = {
            "source": "legacy_builtin",
            "profile": "legacy_builtin",
            "family_exact": {},
            "family_prefix": {},
            "species_token": {"delimiter": ":", "take": "after_first"},
            "metric_family_abs_floor": {},
        }
        eval_cfg["metric_taxonomy_resolved"] = resolved
        return resolved

    path_raw = taxonomy_cfg.get("path", "configs/evaluation/metric_taxonomy_profiles.yaml")
    taxonomy_path = _resolve_path(str(path_raw), base=config_parent)
    if not taxonomy_path.exists():
        if bool(contract.get("enforce", False)):
            raise FileNotFoundError(f"metric taxonomy profile file not found: {taxonomy_path}")
        resolved = {
            "source": "legacy_builtin",
            "profile": "legacy_builtin",
            "family_exact": {},
            "family_prefix": {},
            "species_token": {"delimiter": ":", "take": "after_first"},
            "metric_family_abs_floor": {},
        }
        eval_cfg["metric_taxonomy_resolved"] = resolved
        return resolved
    payload = _load_yaml(taxonomy_path)
    profiles = dict(payload.get("profiles") or {})
    selected = dict(profiles.get(profile) or {})
    if not selected:
        if bool(contract.get("enforce", False)):
            raise ValueError(f"metric taxonomy profile not found: {profile}")
        selected = {}
        profile = "legacy_builtin"
    resolved = {
        "source": "shared_yaml",
        "profile": profile,
        "family_exact": dict(selected.get("family_exact") or {}),
        "family_prefix": dict(selected.get("family_prefix") or {}),
        "species_token": dict(selected.get("species_token") or {"delimiter": ":", "take": "after_first"}),
        "metric_family_abs_floor": dict(selected.get("metric_family_abs_floor") or {}),
    }
    eval_cfg["metric_taxonomy_resolved"] = resolved
    # Preserve config-facing metadata so downstream summaries can report effective profile.
    eval_cfg["metric_taxonomy"] = {
        "source": "shared_yaml",
        "path": str(path_raw),
        "profile": profile,
    }
    return resolved


def _resolve_path(raw: str | Path, *, base: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (base / path).resolve()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_config(cfg: dict[str, Any]) -> str:
    payload = json.dumps(cfg, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_git_commit_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        ).strip()
    except Exception:
        return ""
    return out


class _RuntimeGuard:
    def __init__(
        self,
        *,
        run_id: str,
        run_dir: Path,
        heartbeat_sec: float,
        no_progress_timeout_sec: float,
        max_wallclock_sec: float,
    ) -> None:
        self._run_id = str(run_id)
        self._run_dir = Path(run_dir)
        self._lock_path = self._run_dir / ".run.lock"
        self._heartbeat_sec = float(max(heartbeat_sec, 0.0))
        self._no_progress_timeout_sec = float(max(no_progress_timeout_sec, 0.0))
        self._max_wallclock_sec = float(max(max_wallclock_sec, 0.0))
        self._start_monotonic = pytime.monotonic()
        self._last_progress_monotonic = self._start_monotonic
        self._last_heartbeat_monotonic = 0.0
        self._acquired = False
        self._pid = int(os.getpid())
        self._started_at = _utc_now_iso()

    @property
    def lock_path(self) -> Path:
        return self._lock_path

    def acquire(self) -> None:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(str(self._lock_path), flags, 0o644)
        except FileExistsError:
            raise RuntimeError(
                f"run_id lock already exists: {self._lock_path}. "
                "Another run with the same --run-id may still be active."
            ) from None
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_id": self._run_id,
                    "pid": self._pid,
                    "started_at": self._started_at,
                    "heartbeat_at": self._started_at,
                    "last_progress_reason": "acquire",
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        self._acquired = True
        now = pytime.monotonic()
        self._last_progress_monotonic = now
        self._last_heartbeat_monotonic = now

    def release(self) -> None:
        if not self._acquired:
            return
        try:
            self._lock_path.unlink(missing_ok=True)
        finally:
            self._acquired = False

    def _write_heartbeat(self, reason: str) -> None:
        if not self._acquired:
            return
        heartbeat_at = _utc_now_iso()
        payload = {
            "run_id": self._run_id,
            "pid": self._pid,
            "started_at": self._started_at,
            "heartbeat_at": heartbeat_at,
            "last_progress_reason": str(reason),
        }
        self._lock_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    def mark_progress(self, reason: str) -> None:
        now = pytime.monotonic()
        self._last_progress_monotonic = now
        if self._heartbeat_sec <= 0.0 or (now - self._last_heartbeat_monotonic) >= self._heartbeat_sec:
            self._write_heartbeat(reason)
            self._last_heartbeat_monotonic = now

    def check(self, reason: str) -> None:
        now = pytime.monotonic()
        if self._max_wallclock_sec > 0.0 and (now - self._start_monotonic) > self._max_wallclock_sec:
            raise TimeoutError(
                f"max_wallclock_sec exceeded ({self._max_wallclock_sec:.1f}s): {self._run_id}"
            )
        if self._no_progress_timeout_sec > 0.0 and (now - self._last_progress_monotonic) > self._no_progress_timeout_sec:
            raise TimeoutError(
                f"no_progress_timeout_sec exceeded ({self._no_progress_timeout_sec:.1f}s): {self._run_id}"
            )
        if self._heartbeat_sec > 0.0 and (now - self._last_heartbeat_monotonic) >= self._heartbeat_sec:
            self._write_heartbeat(reason)
            self._last_heartbeat_monotonic = now

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


def _selection_nonpass_priority(selection_cfg: dict[str, Any]) -> str:
    raw = str(selection_cfg.get("nonpass_priority", "structure_then_score")).strip().lower()
    if raw not in {"structure_then_score", "score_only"}:
        return "structure_then_score"
    return raw


def _structure_deficit_score(row: dict[str, Any]) -> float:
    score = max(0.0, -float(row.get("balance_margin", 0.0)))
    if not bool(row.get("floor_passed", True)):
        score += 1.0
    if not bool(row.get("cluster_guard_passed", True)):
        score += 1.0
    if not bool(row.get("physical_gate_passed", True)):
        score += 1.0
    return float(score)


def _selection_quality_score_raw_drift(row: dict[str, Any]) -> float:
    selection_use_raw_drift = bool(row.get("_selection_use_raw_drift", True))
    if not selection_use_raw_drift:
        return float(row.get("selection_score", 0.0))
    raw_drift = float(row.get("metric_drift_raw", row.get("metric_drift_effective", 1.0)))
    cap = float(row.get("_selection_raw_drift_cap", 2.0))
    if not np.isfinite(raw_drift):
        raw_drift = cap
    if not np.isfinite(cap) or cap <= 1.0:
        cap = 2.0
    normalized = float(np.clip((raw_drift - 1.0) / max(cap - 1.0, 1.0e-12), 0.0, 1.0))
    return float(1.0 - normalized)


def _effective_metric_drift(raw_metric_drift: float, *, cap: float) -> float:
    raw = float(raw_metric_drift)
    cap_eff = float(cap)
    if not np.isfinite(raw):
        raw = 1.0
    if (not np.isfinite(cap_eff)) or cap_eff < 1.0:
        cap_eff = 1.30
    return float(np.clip(raw, 1.0, cap_eff))


def _trim_keep_by_reaction_species_ratio(
    keep_reactions: np.ndarray,
    *,
    species_after: int,
    max_reaction_species_ratio: float,
    min_keep_count: int = 1,
    importance: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    keep = np.asarray(keep_reactions, dtype=bool).reshape(-1).copy()
    selected_before = int(np.sum(keep))
    safe_species_after = max(int(species_after), 0)
    min_keep = max(int(min_keep_count), 1)
    if selected_before <= 0 or safe_species_after <= 0:
        return keep, {
            "applied": False,
            "selected_before": selected_before,
            "selected_after": selected_before,
            "allowed_reactions": selected_before,
            "dropped_reactions": 0,
            "ratio_before": 0.0,
            "ratio_after": 0.0,
        }

    max_ratio = float(max_reaction_species_ratio)
    if (not np.isfinite(max_ratio)) or max_ratio <= 0.0:
        return keep, {
            "applied": False,
            "selected_before": selected_before,
            "selected_after": selected_before,
            "allowed_reactions": selected_before,
            "dropped_reactions": 0,
            "ratio_before": float(selected_before) / float(safe_species_after),
            "ratio_after": float(selected_before) / float(safe_species_after),
        }

    allowed = int(np.floor(max_ratio * float(safe_species_after)))
    allowed = max(allowed, min_keep)
    if selected_before <= allowed:
        ratio_now = float(selected_before) / float(safe_species_after)
        return keep, {
            "applied": False,
            "selected_before": selected_before,
            "selected_after": selected_before,
            "allowed_reactions": allowed,
            "dropped_reactions": 0,
            "ratio_before": ratio_now,
            "ratio_after": ratio_now,
        }

    drop_count = int(selected_before - allowed)
    selected_idx = np.flatnonzero(keep)
    if importance is not None and np.asarray(importance).shape == keep.shape:
        importance_vec = np.nan_to_num(np.asarray(importance, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        selected_importance = importance_vec[selected_idx]
    else:
        selected_importance = np.arange(selected_idx.size, dtype=float)
    drop_order = np.argsort(selected_importance, kind="stable")
    drop_idx = selected_idx[drop_order[:drop_count]]
    keep[drop_idx] = False
    selected_after = int(np.sum(keep))
    return keep, {
        "applied": True,
        "selected_before": selected_before,
        "selected_after": selected_after,
        "allowed_reactions": allowed,
        "dropped_reactions": int(selected_before - selected_after),
        "ratio_before": float(selected_before) / float(safe_species_after),
        "ratio_after": float(selected_after) / float(safe_species_after),
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
    nonpass_priority = _selection_nonpass_priority(selection_cfg)
    tie_breakers = [
        str(x).strip()
        for x in list(selection_cfg.get("tie_breakers") or ["reaction_reduction", "species_reduction", "mean_rel_diff"])
        if str(x).strip()
    ]

    for row in stage_rows:
        floors = dict(row.get("_floors") or {})
        row["selection_score"] = _stage_selection_score(row, floors, weights)
        row["structure_deficit_score"] = _structure_deficit_score(row)
        row["selection_quality_score_raw_drift"] = _selection_quality_score_raw_drift(row)

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

    if passed_non_degraded:
        candidate_pool = passed_non_degraded
        selection_pool_kind = "passed"
    elif passed:
        candidate_pool = passed
        selection_pool_kind = "passed"
    elif floor_non_degraded:
        candidate_pool = floor_non_degraded
        selection_pool_kind = "floor"
    elif floor_rows:
        candidate_pool = floor_rows
        selection_pool_kind = "floor"
    elif non_degraded:
        candidate_pool = non_degraded
        selection_pool_kind = "non_degraded"
    else:
        candidate_pool = stage_rows
        selection_pool_kind = "all"
    for row in candidate_pool:
        row["selection_pool_kind"] = str(selection_pool_kind)
    if selection_pool_kind != "passed" and nonpass_priority == "structure_then_score":
        pareto = list(candidate_pool)
    else:
        pareto = _pareto_rows(candidate_pool)
        if not pareto:
            pareto = list(candidate_pool)
    if selection_pool_kind != "passed" and nonpass_priority == "structure_then_score":
        pareto.sort(
            key=lambda r: (
                float(r.get("structure_deficit_score", _structure_deficit_score(r))),
                -float(r.get("selection_score", -1.0e9)),
                -float(r.get("selection_quality_score_raw_drift", _selection_quality_score_raw_drift(r))),
                *_tie_breaker_sort_key(r, tie_breakers),
                int(r.get("reactions_after", 1.0e9)),
                int(r.get("species_after", 1.0e9)),
            )
        )
    else:
        pareto.sort(
            key=lambda r: (
                -float(r.get("selection_score", -1.0e9)),
                -float(r.get("selection_quality_score_raw_drift", _selection_quality_score_raw_drift(r))),
                *_tie_breaker_sort_key(r, tie_breakers),
                int(r.get("reactions_after", 1.0e9)),
                int(r.get("species_after", 1.0e9)),
            )
        )
    selected = pareto[0]
    selected["selection_pool_kind"] = str(selection_pool_kind)
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
    nonpass_priority = _selection_nonpass_priority(selection_cfg)

    if policy == "pass_first_pareto":
        return _select_stage_pass_first(stage_rows, cfg)

    if policy != "physics_first_pareto":
        selected = _choose_stage(stage_rows)
        selected["selection_score"] = _stage_selection_score(selected, dict(selected.get("_floors") or {}), weights)
        selected["structure_deficit_score"] = _structure_deficit_score(selected)
        selected["selection_pool_kind"] = "all"
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

    if passed_floor_non_degraded:
        selection_pool_kind = "passed"
    elif passed_floor:
        selection_pool_kind = "passed"
    elif floor_non_degraded:
        selection_pool_kind = "floor"
    elif floor_rows:
        selection_pool_kind = "floor"
    elif non_degraded:
        selection_pool_kind = "non_degraded"
    else:
        selection_pool_kind = "all"

    candidate_pool = passed_floor_non_degraded or passed_floor or floor_non_degraded or floor_rows or non_degraded or stage_rows
    if selection_pool_kind != "passed" and nonpass_priority == "structure_then_score":
        pareto = list(candidate_pool)
    else:
        pareto = _pareto_rows(candidate_pool)
        if not pareto:
            pareto = list(candidate_pool)

    for row in stage_rows:
        floors = dict(row.get("_floors") or {})
        row["selection_score"] = _stage_selection_score(row, floors, weights)
        row["structure_deficit_score"] = _structure_deficit_score(row)
        row["selection_quality_score_raw_drift"] = _selection_quality_score_raw_drift(row)
    for row in pareto:
        row["selection_pool_kind"] = str(selection_pool_kind)
    if selection_pool_kind != "passed" and nonpass_priority == "structure_then_score":
        pareto.sort(
            key=lambda r: (
                float(r.get("structure_deficit_score", _structure_deficit_score(r))),
                -float(r.get("selection_score", -1.0e9)),
                -float(r.get("selection_quality_score_raw_drift", _selection_quality_score_raw_drift(r))),
                float(r.get("mean_rel_diff", 1.0e9)),
                int(r.get("reactions_after", 1.0e9)),
                int(r.get("species_after", 1.0e9)),
            )
        )
    else:
        pareto.sort(
            key=lambda r: (
                -float(r.get("selection_score", -1.0e9)),
                -float(r.get("selection_quality_score_raw_drift", _selection_quality_score_raw_drift(r))),
                float(r.get("mean_rel_diff", 1.0e9)),
                int(r.get("reactions_after", 1.0e9)),
                int(r.get("species_after", 1.0e9)),
            )
        )
    selected = pareto[0]
    selected["selection_pool_kind"] = str(selection_pool_kind)

    return {"selected": selected, "pareto_candidates": pareto, "weights": weights, "policy": policy}


def _derive_blockers(selected: dict[str, Any]) -> dict[str, Any]:
    validity_failed = not bool(selected.get("mandatory_validity_passed", True))
    error_failed = not bool(selected.get("error_gate_passed", selected.get("gate_passed", False)))
    structure_reasons: list[str] = []
    if int(selected.get("hard_ban_violations", 0)) > 0:
        structure_reasons.append("hard_ban")
    if not bool(selected.get("physical_gate_passed", True)):
        structure_reasons.append("physical_gate")
    if not bool(selected.get("floor_passed", True)):
        structure_reasons.append("physics_floor")
    if not bool(selected.get("balance_gate_passed", True)):
        structure_reasons.append("balance_gate")
    if not bool(selected.get("cluster_guard_passed", True)):
        structure_reasons.append("cluster_guard")
    structure_failed = bool(structure_reasons)
    blockers: list[str] = []
    if validity_failed:
        blockers.append("validity")
    if error_failed:
        blockers.append("error")
    if structure_failed:
        blockers.append("structure")
    primary_blocker_layer = blockers[0] if blockers else "none"
    validity_fail_reason_primary = "mandatory_threshold_not_met" if validity_failed else "none"
    error_fail_reason_primary = (
        str(selected.get("error_fail_reason_primary", "none")) if error_failed else "none"
    )
    secondary_blockers = blockers[1:]
    return {
        "primary_blocker_layer": str(primary_blocker_layer),
        "secondary_blockers": [str(x) for x in secondary_blockers],
        "validity_fail_reason_primary": str(validity_fail_reason_primary),
        "error_fail_reason_primary": str(error_fail_reason_primary),
        "structure_fail_reasons": [str(x) for x in structure_reasons],
    }


def _validate_tiered_error_aggregation_config(
    error_aggregation_cfg: dict[str, Any], *, require_explicit: bool = False
) -> None:
    mode = str(error_aggregation_cfg.get("mode", "tiered")).strip().lower()
    if mode != "tiered":
        raise ValueError(
            "evaluation.error_aggregation.mode must be 'tiered' (legacy modes are removed)"
        )
    explicit = bool(error_aggregation_cfg.get("require_explicit_thresholds", False))
    if require_explicit and not explicit:
        raise ValueError(
            "evaluation.error_aggregation.require_explicit_thresholds must be true"
        )
    if not explicit and not require_explicit:
        return
    required = (
        "mandatory_case_pass_min",
        "optional_metric_pass_min",
        "max_mean_rel_diff_mandatory",
        "max_mean_rel_diff_optional",
    )
    missing = [key for key in required if key not in error_aggregation_cfg]
    if missing:
        raise ValueError(
            "evaluation.error_aggregation missing required keys for mode=tiered: "
            + ", ".join(str(x) for x in missing)
        )


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


def _validate_explicit_adaptive_kfold_policy(split_cfg: dict[str, Any]) -> None:
    if not bool(split_cfg.get("enforce_explicit_policy", False)):
        return
    policy_raw = split_cfg.get("kfold_policy")
    if not isinstance(policy_raw, dict):
        raise ValueError(
            "evaluation.surrogate_split.kfold_policy is required when enforce_explicit_policy=true"
        )
    required = ("min_cases_for_kfold", "default_k", "k_by_case_count")
    missing = [key for key in required if key not in policy_raw]
    if missing:
        raise ValueError(
            "evaluation.surrogate_split.kfold_policy missing required keys: "
            + ", ".join(str(x) for x in missing)
        )
    if not isinstance(policy_raw.get("k_by_case_count"), dict):
        raise ValueError("evaluation.surrogate_split.kfold_policy.k_by_case_count must be a mapping")


def _resolve_surrogate_split_adaptive(conditions: list[dict[str, Any]], eval_cfg: dict[str, Any]) -> dict[str, Any]:
    split_cfg = dict(eval_cfg.get("surrogate_split") or {})
    _validate_explicit_adaptive_kfold_policy(split_cfg)
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
    cost_cfg = dict(split_cfg.get("cost_control") or {})
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
        max_effective_folds = int(cost_cfg.get("max_effective_folds", 0))
        if max_effective_folds > 0:
            k = max(2, min(k, max_effective_folds))
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
    split_cfg = dict(eval_cfg.get("surrogate_split") or {})
    cost_cfg = dict(split_cfg.get("cost_control") or {})
    early_stop_on_stable_fail = bool(cost_cfg.get("early_stop_on_stable_fail", False))
    stable_fail_streak = int(max(1, cost_cfg.get("stable_fail_streak", 2)))
    stable_fail_pass_rate_max = float(cost_cfg.get("stable_fail_pass_rate_max", 0.50))
    stable_fail_mean_rel_min = float(cost_cfg.get("stable_fail_mean_rel_min", float(eval_cfg.get("rel_tolerance", 0.40))))
    baseline_by_case = {str(r.get("case_id")): r for r in baseline_rows}
    cond_by_case = {str(c.get("case_id")): c for c in conditions}

    if surrogate_model_name != "linear_ridge":
        artifact = {"reference_rows": baseline_rows, "global_scale": metric_drift}
        surrogate_rows = run_surrogate_cases(artifact, conditions, qoi_cfg)
        _, summary = compare_with_baseline(baseline_rows, surrogate_rows, eval_cfg, qoi_cfg=qoi_cfg)
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
        _, summary = compare_with_baseline(baseline_rows, surrogate_rows, eval_cfg, qoi_cfg=qoi_cfg)
        return surrogate_rows, summary, []

    folds = list(split_plan.get("folds") or [])
    predicted: list[dict[str, Any]] = []
    baseline_eval: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    stable_fail_hits = 0
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
        _, fold_summary = compare_with_baseline(test_baseline, fold_pred, eval_cfg, qoi_cfg=qoi_cfg)
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
        fold_fail = (
            float(fold_summary.pass_rate) <= stable_fail_pass_rate_max
            and float(fold_summary.mean_rel_diff or 0.0) >= stable_fail_mean_rel_min
        )
        stable_fail_hits = (stable_fail_hits + 1) if fold_fail else 0
        if early_stop_on_stable_fail and stable_fail_hits >= stable_fail_streak:
            break

    if not predicted:
        artifact = {"reference_rows": baseline_rows, "global_scale": metric_drift}
        predicted = run_surrogate_cases(artifact, conditions, qoi_cfg)
        _, summary = compare_with_baseline(baseline_rows, predicted, eval_cfg, qoi_cfg=qoi_cfg)
        return predicted, summary, []

    _, summary = compare_with_baseline(baseline_eval, predicted, eval_cfg, qoi_cfg=qoi_cfg)
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
    projection_start = pytime.perf_counter()
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
    projection_elapsed = float(pytime.perf_counter() - projection_start)
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
        "timing_projection_s": projection_elapsed,
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
    builtin_cfg = dict(qoi_cfg.get("qoi_builtin_metrics") or {})
    include_temperature_metrics = bool(builtin_cfg.get("include_temperature_metrics", True))
    include_ignition_delay = bool(builtin_cfg.get("include_ignition_delay", True))

    idx = {s: i for i, s in enumerate(case.species_names)}
    time = np.asarray(case.time, dtype=float)
    out: dict[str, float] = {}
    if include_ignition_delay:
        out["ignition_delay"] = _ignition_delay(time, case.temperature)
    if include_temperature_metrics:
        out["T_max"] = float(np.max(case.temperature))
        out["T_last"] = float(case.temperature[-1])
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


def _evaluate_metric_replay_health_trust(
    *,
    metric_clip_ratio: float,
    guardrail_trigger_ratio: float,
    max_metric_clip_ratio: float,
    min_guardrail_trigger_ratio: float,
) -> dict[str, Any]:
    clip_exceeded = bool(metric_clip_ratio > max_metric_clip_ratio)
    trigger_reliable = bool(guardrail_trigger_ratio >= min_guardrail_trigger_ratio)
    trust_invalid = bool(clip_exceeded and trigger_reliable)
    return {
        "metric_clip_ratio": float(metric_clip_ratio),
        "metric_clip_guardrail_trigger_ratio": float(guardrail_trigger_ratio),
        "max_metric_clip_ratio": float(max_metric_clip_ratio),
        "min_guardrail_trigger_ratio": float(min_guardrail_trigger_ratio),
        "clip_exceeded": bool(clip_exceeded),
        "trigger_reliable": bool(trigger_reliable),
        "replay_health_trust_invalid": bool(trust_invalid),
    }


def _compute_structure_feedback_multiplier(
    *,
    prev_stage_structure: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> float:
    enabled = bool(cfg.get("enabled", False))
    if not enabled or not prev_stage_structure:
        return 1.0

    alpha = max(float(cfg.get("alpha_domain", 0.8)), 0.0)
    beta = max(float(cfg.get("beta_balance", 0.5)), 0.0)
    cap = max(float(cfg.get("max_multiplier", 1.35)), 1.0)

    domain_deficit = max(float(prev_stage_structure.get("domain_deficit", 0.0) or 0.0), 0.0)
    balance_margin = float(prev_stage_structure.get("balance_margin", 0.0) or 0.0)
    balance_deficit = max(-balance_margin, 0.0)

    mult = 1.0 + (alpha * domain_deficit) + (beta * balance_deficit)
    return float(np.clip(mult, 1.0, cap))


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
    prev_stage_structure: dict[str, Any] | None,
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

    structure_cfg = dict(adaptive_cfg.get("structure_feedback") or {})
    structure_mult = _compute_structure_feedback_multiplier(prev_stage_structure=prev_stage_structure, cfg=structure_cfg)
    adjusted = raw * source_mult * split_mult * stage_mult_val * feedback_mult * physical_mult * structure_mult
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
        "structure_feedback_enabled": bool(structure_cfg.get("enabled", False)),
        "structure_feedback_multiplier": float(structure_mult),
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
    model_cfg = dict(cfg.get("model") or {})
    raw_backend = str(model_cfg.get("backend", "pyg")).strip().lower() or "pyg"
    backend_candidates_cfg = model_cfg.get("backend_candidates")
    backend_candidates: list[str]
    if isinstance(backend_candidates_cfg, (list, tuple)) and backend_candidates_cfg:
        backend_candidates = [str(x).strip().lower() for x in backend_candidates_cfg if str(x).strip()]
    else:
        backend_candidates = [raw_backend, "numpy"]
    normalized_backends: list[str] = []
    seen_backends: set[str] = set()
    for backend in backend_candidates:
        if backend in seen_backends:
            continue
        seen_backends.add(backend)
        normalized_backends.append(backend)

    candidate_cfg = dict(cfg.get("candidate_selection") or {})
    candidate_selection_enabled = bool(candidate_cfg.get("enabled", True))
    dedupe_by_hash = bool(candidate_cfg.get("dedupe_by_assignment_hash", True))
    swap_cfg = dict(candidate_cfg.get("fallback_swap_refine") or {})
    swap_enabled = bool(swap_cfg.get("enabled", True))
    swap_max_steps = int(swap_cfg.get("max_steps", 32))
    swap_min_cov_improve = float(swap_cfg.get("min_coverage_improve", 1.0e-6))
    max_cluster_size_ratio = float(train_cfg.get("max_cluster_size_ratio", 1.0))
    species_weights = np.linalg.norm(features, axis=1)

    def _candidate_dynamics_recon_error(S_cand: np.ndarray) -> float:
        s_mat = np.asarray(S_cand, dtype=float)
        if wdot_arr.size == 0 or s_mat.size == 0:
            return 0.0
        try:
            pinv = np.linalg.pinv(s_mat)
        except np.linalg.LinAlgError:
            return float("inf")
        wdot_reduced = np.asarray(wdot_arr @ s_mat, dtype=float)
        wdot_recon = np.asarray(wdot_reduced @ pinv, dtype=float)
        denom = np.maximum(np.abs(wdot_arr), 1.0e-9)
        rel = np.abs(wdot_recon - wdot_arr) / denom
        return float(np.mean(rel))

    trained_candidates: list[dict[str, Any]] = []
    for backend in normalized_backends:
        cfg_backend = dict(cfg)
        model_cfg_backend = dict(model_cfg)
        model_cfg_backend["backend"] = backend
        model_cfg_backend["n_clusters"] = int(max(1, round(features.shape[0] * float(target_ratio))))
        cfg_backend["model"] = model_cfg_backend
        trained = train_pooling_assignment(graph, features, constraints, cfg_backend)
        S = np.asarray(trained.get("S"), dtype=float)
        assign_hash = hashlib.sha256((S > 0.5).astype(np.uint8).tobytes()).hexdigest()
        tm = dict(trained.get("train_metrics") or {})
        trained_candidates.append(
            {
                "backend": backend,
                "source": "backend",
                "trained": trained,
                "S": S,
                "assignment_hash": assign_hash,
                "coverage_proxy": float(tm.get("coverage_proxy", 0.0)),
                "max_cluster_size_ratio": float(tm.get("max_cluster_size_ratio", 1.0)),
                "constraint_loss": float(tm.get("constraint_loss", 0.0)),
                "hard_ban_violations": int(tm.get("hard_ban_violations", 0)),
                "cluster_guard_passed": bool(tm.get("cluster_guard_passed", True)),
                "dynamics_recon_error": _candidate_dynamics_recon_error(S),
            }
        )

    unique_candidates: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for cand in trained_candidates:
        if dedupe_by_hash and cand["assignment_hash"] in seen_hashes:
            continue
        seen_hashes.add(cand["assignment_hash"])
        unique_candidates.append(cand)
    if not unique_candidates:
        unique_candidates = list(trained_candidates)

    if candidate_selection_enabled and len(unique_candidates) == 1 and swap_enabled:
        base = unique_candidates[0]
        swap_result = _refine_cluster_balance_swap(
            S=np.asarray(base.get("S"), dtype=float),
            hard_mask=hard_mask,
            pair_cost=pair_cost,
            species_weights=species_weights,
            max_cluster_size_ratio=max_cluster_size_ratio,
            max_steps=swap_max_steps,
            min_coverage_improve=swap_min_cov_improve,
        )
        if bool(swap_result.get("improved", False)):
            s_swap = np.asarray(swap_result.get("S"), dtype=float)
            assignment_hash = hashlib.sha256((s_swap > 0.5).astype(np.uint8).tobytes()).hexdigest()
            if assignment_hash != base.get("assignment_hash"):
                trained_base = dict(base.get("trained") or {})
                train_metrics = dict((trained_base.get("train_metrics") or {}))
                train_metrics["coverage_proxy"] = float(swap_result.get("coverage_proxy_after", train_metrics.get("coverage_proxy", 0.0)))
                train_metrics["max_cluster_size_ratio"] = float(
                    swap_result.get("max_cluster_size_ratio_after", train_metrics.get("max_cluster_size_ratio", 1.0))
                )
                train_metrics["cluster_guard_passed"] = bool(
                    train_metrics.get("max_cluster_size_ratio", 1.0) <= train_metrics.get("max_cluster_size_ratio_limit", 1.0) + 1.0e-12
                )
                train_metrics["constraint_loss"] = float(
                    swap_result.get("constraint_loss_after", train_metrics.get("constraint_loss", 0.0))
                )
                trained_base["S"] = s_swap
                trained_base["S_prob"] = np.asarray(s_swap, dtype=float)
                trained_base["train_metrics"] = train_metrics
                unique_candidates.append(
                    {
                        "backend": str(base.get("backend", raw_backend)),
                        "source": "swap_refine",
                        "trained": trained_base,
                        "S": s_swap,
                        "assignment_hash": assignment_hash,
                        "coverage_proxy": float(train_metrics.get("coverage_proxy", 0.0)),
                        "max_cluster_size_ratio": float(train_metrics.get("max_cluster_size_ratio", 1.0)),
                        "constraint_loss": float(train_metrics.get("constraint_loss", 0.0)),
                        "hard_ban_violations": int(train_metrics.get("hard_ban_violations", 0)),
                        "cluster_guard_passed": bool(train_metrics.get("cluster_guard_passed", True)),
                        "dynamics_recon_error": _candidate_dynamics_recon_error(s_swap),
                    }
                )

    def _candidate_score(cand: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
        backend_name = str(cand.get("backend", ""))
        backend_pref = 1.0 if backend_name in {"pyg", "torch_geometric", "tgp"} else 0.0
        return (
            1.0 if int(cand.get("hard_ban_violations", 1)) == 0 else 0.0,
            1.0 if bool(cand.get("cluster_guard_passed", False)) else 0.0,
            -float(cand.get("dynamics_recon_error", float("inf"))),
            float(cand.get("coverage_proxy", 0.0)),
            -float(cand.get("max_cluster_size_ratio", 1.0e9)),
            -float(cand.get("constraint_loss", 1.0e9)),
            backend_pref,
        )

    base_candidate = unique_candidates[0]
    selected_candidate = max(unique_candidates, key=_candidate_score)
    if float(selected_candidate.get("dynamics_recon_error", float("inf"))) > float(
        base_candidate.get("dynamics_recon_error", float("inf"))
    ) + 1.0e-12:
        selected_candidate = base_candidate
    if float(selected_candidate.get("coverage_proxy", 0.0)) + 1.0e-12 < float(base_candidate.get("coverage_proxy", 0.0)):
        selected_candidate = base_candidate

    trained_selected = dict(selected_candidate.get("trained") or {})
    S = np.asarray(trained_selected.get("S"), dtype=float)
    cluster_meta = list(trained_selected.get("cluster_meta") or [])
    train_metrics = dict(trained_selected.get("train_metrics") or {})
    hard_viol = int(train_metrics.get("hard_ban_violations", 0))

    mapping = ReductionMapping(
        S=S,
        pool_meta=cluster_meta,
        keep_reactions=None,
        meta={
            "target_ratio": float(target_ratio),
            "achieved_ratio": float(S.shape[1]) / float(max(S.shape[0], 1)),
            "hard_ban_violations": int(hard_viol),
            "pooling_model": str((trained_selected.get("model_info") or {}).get("model_type", "unknown")),
            "pooling_graph": str((trained_selected.get("model_info") or {}).get("graph_type", graph_kind)),
        },
    )

    candidate_scores = [
        {
            "backend": str(cand.get("backend", "")),
            "source": str(cand.get("source", "backend")),
            "assignment_hash": str(cand.get("assignment_hash", "")),
            "coverage_proxy": float(cand.get("coverage_proxy", 0.0)),
            "max_cluster_size_ratio": float(cand.get("max_cluster_size_ratio", 0.0)),
            "constraint_loss": float(cand.get("constraint_loss", 0.0)),
            "hard_ban_violations": int(cand.get("hard_ban_violations", 0)),
            "cluster_guard_passed": bool(cand.get("cluster_guard_passed", True)),
            "dynamics_recon_error": float(cand.get("dynamics_recon_error", 0.0)),
        }
        for cand in unique_candidates
    ]

    saved_path: Path | None = None
    if artifact_path is not None:
        saved_path = save_pooling_artifact(
            artifact_path,
            {
                "S": S,
                "S_prob": np.asarray(trained_selected.get("S_prob"), dtype=float),
                "cluster_meta": cluster_meta,
                "train_metrics": train_metrics,
                "model_info": dict(trained_selected.get("model_info") or {}),
            },
        )

    return mapping, {
        "train_metrics": train_metrics,
        "model_info": dict(trained_selected.get("model_info") or {}),
        "graph_kind": graph_kind,
        "hard_ban_violations": int(hard_viol),
        "candidate_count": int(len(trained_candidates)),
        "candidate_unique_count": int(len(unique_candidates)),
        "candidate_selected_backend": str(selected_candidate.get("backend", raw_backend)),
        "candidate_selected_source": str(selected_candidate.get("source", "backend")),
        "candidate_selected_coverage_proxy": float(selected_candidate.get("coverage_proxy", 0.0)),
        "candidate_selected_dynamics_recon_error": float(
            selected_candidate.get("dynamics_recon_error", 0.0)
        ),
        "candidate_selected_max_cluster_size_ratio": float(
            selected_candidate.get("max_cluster_size_ratio", train_metrics.get("max_cluster_size_ratio", 0.0))
        ),
        "candidate_scores": candidate_scores,
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
    eval_cfg = dict(cfg.get("evaluation") or {"rel_tolerance": 0.4, "rel_eps": 1.0e-12})
    contract_cfg = _resolve_evaluation_contract(eval_cfg)
    non_regression_cfg = dict(eval_cfg.get("non_regression") or {})
    non_regression_cfg.setdefault("reaction_reduction_priority", True)
    non_regression_cfg.setdefault("max_reaction_reduction_drop_ratio", 0.02)
    eval_cfg["non_regression"] = non_regression_cfg
    validity_cfg = dict(eval_cfg.get("gate_metric_validity") or {})
    validity_cfg.setdefault("mandatory_hard_mode", "min_valid_count")
    validity_cfg.setdefault("min_valid_mandatory_count_abs", 1)
    validity_cfg.setdefault("min_valid_mandatory_ratio", 1.0)
    validity_cfg.setdefault("min_valid_mandatory_cap_by_total", True)
    validity_cfg.setdefault("mandatory_metric_validity_mode", "case_pass_rate")
    validity_cfg.setdefault("mandatory_metric_case_pass_min", None)
    validity_cfg.setdefault("validity_scope", "coverage_only")
    validity_cfg.setdefault("mandatory_validity_basis", "coverage_evaluable")
    validity_cfg.setdefault("mandatory_valid_unit_mode", "species_family_quorum")
    validity_cfg.setdefault("mandatory_species_family_score_mode", "uniform")
    validity_cfg.setdefault("mandatory_species_family_case_pass_min", 0.67)
    validity_cfg.setdefault("mandatory_gate_unit_min_evaluable_case_ratio_shadow", 0.25)
    eval_cfg["gate_metric_validity"] = validity_cfg
    error_aggregation_cfg = dict(eval_cfg.get("error_aggregation") or {})
    error_aggregation_cfg.setdefault("mandatory_error_include_validity", False)
    error_aggregation_cfg.setdefault("mandatory_quality_scope", "valid_only")
    error_aggregation_cfg.setdefault("mandatory_tail_scope", "quality_scope")
    error_aggregation_cfg.setdefault("mandatory_tail_guard_mode", "p95")
    error_aggregation_cfg.setdefault("mandatory_tail_guard_policy", "conditional_hard")
    error_aggregation_cfg.setdefault("mandatory_tail_activation_ratio_min", 0.10)
    error_aggregation_cfg.setdefault("mandatory_tail_exceed_ref", "tail_max")
    error_aggregation_cfg.setdefault("mandatory_tail_rel_diff_max", 1.50)
    error_aggregation_cfg.setdefault("mandatory_tail_min_samples", 8)
    eval_cfg["error_aggregation"] = error_aggregation_cfg
    surrogate_drift_cfg = dict(eval_cfg.get("surrogate_drift") or {})
    surrogate_drift_cfg.setdefault("selection_use_raw_drift", True)
    surrogate_drift_cfg.setdefault("raw_drift_cap_for_selection", 2.0)
    surrogate_drift_cfg.setdefault("keep_effective_drift_cap_for_eval", 1.30)
    eval_cfg["surrogate_drift"] = surrogate_drift_cfg
    metric_taxonomy_resolved = _load_metric_taxonomy_profile(
        eval_cfg,
        config_parent=config_path.parent,
        contract=contract_cfg,
    )
    _validate_evaluation_contract(eval_cfg, contract_cfg)
    qoi_builtin_cfg = dict(eval_cfg.get("qoi_builtin_metrics") or {})
    merged_builtin_cfg = {
        "include_temperature_metrics": bool(qoi_builtin_cfg.get("include_temperature_metrics", True)),
        "include_ignition_delay": bool(qoi_builtin_cfg.get("include_ignition_delay", True)),
    }
    qoi_cfg["qoi_builtin_metrics"] = merged_builtin_cfg
    qoi_species_integral = [str(x) for x in list(qoi_cfg.get("species_integral") or []) if str(x)]
    qoi_deposition_integral = [str(x) for x in list(qoi_cfg.get("deposition_integral") or []) if str(x)]
    qoi_integral_keys = [f"X_int:{sp}" for sp in qoi_species_integral] + [f"dep_int:{sp}" for sp in qoi_deposition_integral]
    qoi_integral_count = int(len(qoi_integral_keys))
    runtime_cfg = dict(cfg.get("runtime_control") or {})
    runtime_cfg.update(dict(eval_cfg.get("runtime_control") or {}))
    heartbeat_sec = float(runtime_cfg.get("heartbeat_sec", 10.0))
    no_progress_timeout_sec = float(runtime_cfg.get("no_progress_timeout_sec", 0.0))
    max_wallclock_sec = float(runtime_cfg.get("max_wallclock_sec", 0.0))
    report_root = Path(cfg.get("report_dir", "reports")).resolve()
    run_dir = report_root / args.run_id
    runtime_guard = _RuntimeGuard(
        run_id=args.run_id,
        run_dir=run_dir,
        heartbeat_sec=heartbeat_sec,
        no_progress_timeout_sec=no_progress_timeout_sec,
        max_wallclock_sec=max_wallclock_sec,
    )
    try:
        runtime_guard.acquire()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    atexit.register(runtime_guard.release)
    runtime_guard.mark_progress("start")

    run_started_at = _utc_now_iso()
    total_started = pytime.perf_counter()
    runtime_meta = {
        "config_hash": _hash_config(cfg),
        "git_commit": _resolve_git_commit_short(),
        "pid": int(os.getpid()),
        "started_at": run_started_at,
    }

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
    runtime_guard.mark_progress("input_context_ready")

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
    pooling_bridge_cfg = dict(pooling_cfg.get("bridge") or {})
    pooling_bridge_enabled = bool(pooling_bridge_cfg.get("enable", True))
    pooling_bridge_mode_cfg = str(pooling_bridge_cfg.get("mode", "full")).strip().lower() or "full"
    if pooling_bridge_mode_cfg not in {"full", "light"}:
        pooling_bridge_mode_cfg = "full"
    if not pooling_bridge_enabled:
        pooling_bridge_mode_cfg = "light"
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
    learnckpp_features_cache = (
        _build_learnckpp_features(time=time, case_slices=case_slices, conditions=conditions)
        if reduction_mode in {"learnckpp", "pooling"}
        else None
    )

    policy = _merge_policy_from_config(cfg)
    A = _build_element_matrix(species_meta)

    gate_cfg = dict(cfg.get("gate") or {})
    min_pass_rate = float(gate_cfg.get("min_pass_rate", 0.75))
    max_mean_rel = float(gate_cfg.get("max_mean_rel_diff", 0.40))
    max_mean_rel_guard = float(gate_cfg.get("max_mean_rel_diff_unweighted_guard", max_mean_rel))
    max_cons = float(gate_cfg.get("max_conservation_violation", 0.0))
    error_aggregation_cfg = dict(eval_cfg.get("error_aggregation") or {})
    error_aggregation_cfg.setdefault("mode", "tiered")
    error_aggregation_cfg.setdefault("require_explicit_thresholds", False)
    _validate_tiered_error_aggregation_config(
        error_aggregation_cfg,
        require_explicit=bool(contract_cfg.get("enforce", False)),
    )
    if not bool(error_aggregation_cfg.get("require_explicit_thresholds", False)):
        error_aggregation_cfg.setdefault("mandatory_case_pass_min", min_pass_rate)
        error_aggregation_cfg.setdefault("optional_metric_pass_min", min_pass_rate)
        error_aggregation_cfg.setdefault("max_mean_rel_diff_mandatory", max_mean_rel)
        error_aggregation_cfg.setdefault("max_mean_rel_diff_optional", max_mean_rel_guard)
    error_aggregation_cfg.setdefault("mandatory_case_mode", "ratio_mean")
    error_aggregation_cfg.setdefault("mandatory_case_unit_weight_mode", "uniform")
    error_aggregation_cfg.setdefault("mandatory_quality_scope", "valid_only")
    error_aggregation_cfg.setdefault("mandatory_tail_scope", "quality_scope")
    error_aggregation_cfg.setdefault("mandatory_mean_aggregation", "raw")
    error_aggregation_cfg.setdefault("mandatory_family_weights", {})
    error_aggregation_cfg.setdefault("mandatory_mean_mode", "winsorized")
    error_aggregation_cfg.setdefault("mandatory_winsor_cap_multiplier", 3.0)
    error_aggregation_cfg.setdefault("mandatory_outlier_multiplier", 5.0)
    error_aggregation_cfg.setdefault("mandatory_outlier_ratio_max", 0.20)
    error_aggregation_cfg.setdefault("mandatory_error_include_validity", False)
    error_aggregation_cfg.setdefault("mandatory_tail_guard_policy", "conditional_hard")
    error_aggregation_cfg.setdefault("mandatory_tail_activation_ratio_min", 0.10)
    error_aggregation_cfg.setdefault("mandatory_tail_exceed_ref", "tail_max")
    error_aggregation_cfg.setdefault("mandatory_tail_guard_mode", "p95")
    error_aggregation_cfg.setdefault("mandatory_tail_rel_diff_max", 1.50)
    error_aggregation_cfg.setdefault("mandatory_tail_min_samples", 8)
    error_aggregation_cfg.setdefault("optional_weight", 0.35)
    eval_cfg["error_aggregation"] = error_aggregation_cfg
    denominator_mode = str((eval_cfg.get("metric_normalization") or {}).get("denominator_mode", "max_abs_or_floor")).strip().lower()
    if denominator_mode != "max_abs_or_floor":
        raise ValueError(
            "evaluation.metric_normalization.denominator_mode must be 'max_abs_or_floor'"
        )
    validity_basis = str(validity_cfg.get("mandatory_validity_basis", "coverage_evaluable")).strip().lower()
    if validity_basis != "coverage_evaluable":
        raise ValueError(
            "evaluation.gate_metric_validity.mandatory_validity_basis must be 'coverage_evaluable'"
        )
    surrogate_drift_cfg = dict(eval_cfg.get("surrogate_drift") or {})
    surrogate_drift_selection_use_raw = bool(surrogate_drift_cfg.get("selection_use_raw_drift", True))
    surrogate_drift_raw_cap_for_selection = float(
        surrogate_drift_cfg.get("raw_drift_cap_for_selection", 2.0) or 2.0
    )
    if (not np.isfinite(surrogate_drift_raw_cap_for_selection)) or surrogate_drift_raw_cap_for_selection <= 1.0:
        surrogate_drift_raw_cap_for_selection = 2.0
    surrogate_drift_effective_cap_for_eval = float(
        surrogate_drift_cfg.get("keep_effective_drift_cap_for_eval", 1.30) or 1.30
    )
    if (not np.isfinite(surrogate_drift_effective_cap_for_eval)) or surrogate_drift_effective_cap_for_eval < 1.0:
        surrogate_drift_effective_cap_for_eval = 1.30
    compression_opt_cfg = dict(eval_cfg.get("compression_optimizer") or {})
    compression_optimizer_enabled = bool(compression_opt_cfg.get("enabled", False))
    compression_optimizer_mode = str(compression_opt_cfg.get("mode", "deterministic_grid") or "deterministic_grid")
    compression_extra_trials = int(compression_opt_cfg.get("per_stage_extra_trials", 2) or 2)
    compression_extra_trials = max(0, min(compression_extra_trials, 4))
    compression_reaction_priority = bool(compression_opt_cfg.get("reaction_priority", True))
    compression_guard_mandatory_mean_delta = float(
        compression_opt_cfg.get("max_allowed_mandatory_mean_delta", 1.0e-6) or 1.0e-6
    )
    compression_guard_optional_mean_delta = float(
        compression_opt_cfg.get("max_allowed_optional_mean_delta", 1.0e-6) or 1.0e-6
    )
    compression_guard_mandatory_pass_drop = float(
        compression_opt_cfg.get("max_allowed_mandatory_pass_rate_drop", 0.0) or 0.0
    )
    compression_require_gate_passed = bool(compression_opt_cfg.get("require_gate_passed", True))
    compression_require_structure_passed = bool(compression_opt_cfg.get("require_structure_passed", True))
    eval_cfg["compression_optimizer"] = {
        "enabled": bool(compression_optimizer_enabled),
        "mode": compression_optimizer_mode,
        "per_stage_extra_trials": int(compression_extra_trials),
        "reaction_priority": bool(compression_reaction_priority),
        "max_allowed_mandatory_mean_delta": float(compression_guard_mandatory_mean_delta),
        "max_allowed_optional_mean_delta": float(compression_guard_optional_mean_delta),
        "max_allowed_mandatory_pass_rate_drop": float(compression_guard_mandatory_pass_drop),
        "require_gate_passed": bool(compression_require_gate_passed),
        "require_structure_passed": bool(compression_require_structure_passed),
    }
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
    prev_stage_structure: dict[str, Any] | None = None
    pooling_stage_artifacts: dict[str, str] = {}
    pooling_stage_metrics: dict[str, dict[str, Any]] = {}
    pooling_artifact_root = Path(str(pooling_cfg.get("artifact_dir", "artifacts/pooling"))) / args.run_id
    timing_stage_s: dict[str, float] = {}
    timing_pooling_fit_s: dict[str, float] = {}
    timing_bridge_s: dict[str, float] = {}
    timing_surrogate_eval_s: dict[str, float] = {}
    timing_physical_gate_s: dict[str, float] = {}
    timing_projection_s: dict[str, float] = {}
    compression_trial_keep_scales = (0.90, 0.80, 0.70, 0.60)

    def _run_baseline_stage(
        *,
        stage_idx: int,
        stage_metric_drift: float,
        mapping: Any,
        prune_lambda: float,
        prune_threshold: float,
        prune_keep_ratio: float,
        prune_exact: bool,
        max_reaction_species_ratio: float,
        floor_min_reactions: int,
    ) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]], Any, dict[str, Any], int, int, float, list[dict[str, Any]], dict[str, float]]:
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
        keep_local, ratio_trim_meta = _trim_keep_by_reaction_species_ratio(
            np.asarray(keep_local, dtype=bool),
            species_after=int(np.asarray(mapping.S).shape[1]),
            max_reaction_species_ratio=float(max_reaction_species_ratio),
            min_keep_count=int(floor_min_reactions),
            importance=np.asarray(i_reaction, dtype=float),
        )
        prune_local = dict(prune_local)
        prune_local["balance_ratio_trim"] = ratio_trim_meta
        if bool(ratio_trim_meta.get("applied", False)):
            prune_local["status"] = f"{prune_local.get('status', 'ok')}:ratio_trimmed"
        t_surrogate_start = pytime.perf_counter()
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
        surrogate_elapsed = float(pytime.perf_counter() - t_surrogate_start)
        t_physical_start = pytime.perf_counter()
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
        physical_elapsed = float(pytime.perf_counter() - t_physical_start)
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
            {
                "timing_surrogate_eval_s": surrogate_elapsed,
                "timing_physical_gate_s": physical_elapsed,
                "timing_projection_s": float(physical_local.get("timing_projection_s", 0.0)),
            },
        )

    for stage_idx, stage in enumerate(stages):
        runtime_guard.check(f"stage_{stage_idx}_start")
        stage_started = pytime.perf_counter()
        name = str(stage.get("name", "unknown"))
        runtime_guard.mark_progress(f"stage_{name}_entered")
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
        stage_pooling_fit_elapsed_s = 0.0
        stage_bridge_elapsed_s = 0.0
        stage_surrogate_elapsed_s = 0.0
        stage_physical_elapsed_s = 0.0
        stage_projection_elapsed_s = 0.0
        if reduction_mode == "pooling":
            t_pooling_fit_start = pytime.perf_counter()
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
                    "candidate_count": 0,
                    "candidate_unique_count": 0,
                    "candidate_selected_backend": "",
                    "candidate_selected_source": "backend",
                    "candidate_selected_coverage_proxy": 0.0,
                    "candidate_selected_dynamics_recon_error": 0.0,
                    "candidate_selected_max_cluster_size_ratio": 0.0,
                    "candidate_scores": [],
                }
                pooling_mapping_fallback_reasons.append({"stage": name, "reason": stage_pooling_metrics["error"]})
            stage_pooling_fit_elapsed_s = float(pytime.perf_counter() - t_pooling_fit_start)
            runtime_guard.mark_progress(f"stage_{name}_pooling_fit_done")
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
        stage_metric_drift_raw = float(metric_drift)
        stage_metric_drift = _effective_metric_drift(
            stage_metric_drift_raw, cap=surrogate_drift_effective_cap_for_eval
        )
        stage_reduction_mode = reduction_mode
        learnckpp_fallback_reason: str | None = None
        stage_pooling_fallback_reason: str | None = None
        learnckpp_target_keep_ratio = prune_keep_ratio
        learnckpp_keep_ratio_policy: dict[str, Any] = {}
        structure_feedback_multiplier = 1.0
        compression_refine_applied = False
        compression_refine_trials = 0
        compression_refine_reaction_delta = 0
        compression_refine_species_delta = 0
        compression_refine_mode_effective = "none"
        compression_refine_guard_passed = True
        nu_balance = np.zeros((int(mapping.S.shape[1]), 0), dtype=float)
        stage_pooling_bridge_mode = pooling_bridge_mode_cfg if reduction_mode == "pooling" else "full"
        runtime_guard.check(f"stage_{name}_pre_eval")

        if reduction_mode in {"learnckpp", "pooling"}:
            t_bridge_start = pytime.perf_counter()
            try:
                ydot_target = np.asarray(wdot @ np.asarray(mapping.S, dtype=float), dtype=float)
                if learnckpp_features_cache is None:
                    learnckpp_features = _build_learnckpp_features(time=time, case_slices=case_slices, conditions=conditions)
                else:
                    learnckpp_features = np.asarray(learnckpp_features_cache, dtype=float)

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
                    prev_stage_structure=prev_stage_structure,
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
                structure_feedback_multiplier = float(keep_policy.get("structure_feedback_multiplier", 1.0) or 1.0)
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
                    light_pooling_bridge = reduction_mode == "pooling" and stage_pooling_bridge_mode == "light"
                    if light_pooling_bridge:
                        y_pred = np.asarray(X @ np.asarray(mapping.S, dtype=float), dtype=float)
                        rates_pred = np.asarray(rates_target_sel, dtype=float)
                        if rates_pred.shape[0] != ydot_target.shape[0]:
                            rates_pred = np.zeros((ydot_target.shape[0], nu_sel.shape[1]), dtype=float)
                        select_status = f"{str(select_details.get('status'))}:light_bridge"
                    else:
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
                        y_ref = np.asarray(X @ np.asarray(mapping.S, dtype=float), dtype=float)
                        rel = np.abs(y_pred - y_ref) / (np.abs(y_ref) + 1.0e-12)
                        stage_metric_drift_raw = float(1.0 + float(np.mean(rel)))
                        stage_metric_drift = _effective_metric_drift(
                            stage_metric_drift_raw, cap=surrogate_drift_effective_cap_for_eval
                        )
                        select_status = str(select_details.get("status"))

                overall_selected = int(np.sum(keep)) if keep.size else 0
                overall_select_ratio = float(overall_selected / max(overall_candidates, 1))
                prune_details = {
                    "status": select_status,
                    "keep_count": overall_selected,
                    "keep_ratio": overall_select_ratio,
                    "mode": "learnckpp",
                }
                post_prune_meta = dict(select_details.get("post_prune_refine") or {})
                post_prune_enabled = bool(post_prune_meta.get("enabled", False))
                compression_refine_mode_effective = "postselect_refine" if post_prune_enabled else "none"
                compression_refine_applied = bool(post_prune_meta.get("applied", False))
                compression_refine_trials = int(post_prune_meta.get("steps", 0) or 0) if post_prune_enabled else 0
                compression_refine_reaction_delta = int(post_prune_meta.get("dropped_total", 0) or 0)
                compression_refine_species_delta = int(0)
                compression_refine_guard_passed = True
                prune_details["post_prune_refine"] = post_prune_meta

                t_surrogate_start = pytime.perf_counter()
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
                stage_surrogate_elapsed_s = float(pytime.perf_counter() - t_surrogate_start)

                keep_gate = np.ones((int(nu_sel.shape[1]),), dtype=bool)
                t_physical_start = pytime.perf_counter()
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
                stage_physical_elapsed_s = float(pytime.perf_counter() - t_physical_start)
                stage_projection_elapsed_s = float(physical_result.get("timing_projection_s", 0.0))
                stage_bridge_elapsed_s = float(pytime.perf_counter() - t_bridge_start)
                runtime_guard.mark_progress(f"stage_{name}_bridge_done")
            except Exception as exc:
                stage_bridge_elapsed_s = float(pytime.perf_counter() - t_bridge_start)
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
                    stage_timing_local,
                ) = _run_baseline_stage(
                    stage_idx=stage_idx,
                    stage_metric_drift=metric_drift,
                    mapping=mapping,
                    prune_lambda=prune_lambda,
                    prune_threshold=prune_threshold,
                    prune_keep_ratio=prune_keep_ratio,
                    prune_exact=prune_exact,
                    max_reaction_species_ratio=float(balance_bands.get("max_reaction_species_ratio", 1.0e9)),
                    floor_min_reactions=int(stage_floor_min_reactions),
                )
                fallback_prefix = ("pooling" if reduction_mode == "pooling" else "learnckpp")
                prune_details["status"] = f"{fallback_prefix}_failed_baseline:{prune_details.get('status')}"
                learnckpp_target_keep_ratio = prune_keep_ratio
                learnckpp_keep_ratio_policy = {"enabled": False, "fallback": True}
                nu_balance = np.asarray(np.asarray(mapping.S, dtype=float).T @ np.asarray(nu[:, keep], dtype=float), dtype=float)
                stage_surrogate_elapsed_s = float(stage_timing_local.get("timing_surrogate_eval_s", 0.0))
                stage_physical_elapsed_s = float(stage_timing_local.get("timing_physical_gate_s", 0.0))
                stage_projection_elapsed_s = float(stage_timing_local.get("timing_projection_s", 0.0))
                runtime_guard.mark_progress(f"stage_{name}_baseline_fallback_done")
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
                stage_timing_local,
            ) = _run_baseline_stage(
                stage_idx=stage_idx,
                stage_metric_drift=stage_metric_drift,
                mapping=mapping,
                prune_lambda=prune_lambda,
                prune_threshold=prune_threshold,
                prune_keep_ratio=prune_keep_ratio,
                prune_exact=prune_exact,
                max_reaction_species_ratio=float(balance_bands.get("max_reaction_species_ratio", 1.0e9)),
                floor_min_reactions=int(stage_floor_min_reactions),
            )
            nu_balance = np.asarray(np.asarray(mapping.S, dtype=float).T @ np.asarray(nu[:, keep], dtype=float), dtype=float)
            stage_surrogate_elapsed_s = float(stage_timing_local.get("timing_surrogate_eval_s", 0.0))
            stage_physical_elapsed_s = float(stage_timing_local.get("timing_physical_gate_s", 0.0))
            stage_projection_elapsed_s = float(stage_timing_local.get("timing_projection_s", 0.0))
            runtime_guard.mark_progress(f"stage_{name}_baseline_done")

        if (
            compression_optimizer_enabled
            and compression_extra_trials > 0
            and compression_optimizer_mode == "deterministic_grid"
            and stage_reduction_mode in {"baseline", "baseline_fallback"}
        ):
            keep_base = np.asarray(keep, dtype=bool).reshape(-1)
            base_reactions_after = int(np.sum(keep_base))
            trial_scales = compression_trial_keep_scales[:compression_extra_trials]
            compression_refine_trials = int(len(trial_scales))
            compression_refine_mode_effective = "baseline_grid"

            base_pass_rate_mandatory = float(
                getattr(eval_summary, "pass_rate_mandatory_case", eval_summary.pass_rate) or 0.0
            )
            base_mean_rel_mandatory = float(
                getattr(eval_summary, "mean_rel_diff_mandatory", eval_summary.mean_rel_diff or 0.0) or 0.0
            )
            base_mean_rel_optional = float(
                getattr(eval_summary, "mean_rel_diff_optional", eval_summary.mean_rel_diff or 0.0) or 0.0
            )
            base_coverage_gate_passed = bool(
                getattr(eval_summary, "coverage_gate_passed", getattr(eval_summary, "mandatory_validity_passed", True))
            )
            base_error_gate_passed = bool(getattr(eval_summary, "error_gate_passed", True))
            base_gate_passed = bool(base_coverage_gate_passed and base_error_gate_passed)
            base_quality_guard_passed = bool(
                base_pass_rate_mandatory >= (base_pass_rate_mandatory - compression_guard_mandatory_pass_drop)
            )
            compression_refine_guard_passed = bool(base_quality_guard_passed)
            if compression_require_gate_passed and not base_gate_passed:
                compression_refine_guard_passed = False

            if (
                compression_refine_guard_passed
                and base_reactions_after > int(stage_floor_min_reactions)
                and keep_base.shape[0] == int(n_reactions)
            ):
                s_arr_refine = np.asarray(mapping.S, dtype=float)
                x_reduced_refine = np.asarray(X @ s_arr_refine, dtype=float)
                wdot_reduced_refine = np.asarray(wdot @ s_arr_refine, dtype=float)
                f_reduced_refine = np.asarray(s_arr_refine.T @ np.asarray(F_bar, dtype=float) @ s_arr_refine, dtype=float)
                balance_activity_weights = _build_species_activity_weights(
                    x_reduced_refine,
                    wdot_reduced_refine,
                    f_reduced_refine,
                    cfg,
                )
                balance_essential_mask = _build_essential_cluster_mask(s_arr_refine, species_meta, essential_species)
                floor_min_species_local = int(floors.get("min_species_after", 0))
                floor_min_reactions_local = int(floors.get("min_reactions_after", 0))
                best_trial: dict[str, Any] | None = None
                trial_timing_physical = 0.0
                trial_timing_projection = 0.0

                keep_indices = [int(i) for i in np.where(keep_base)[0]]
                sorted_keep_indices = sorted(keep_indices, key=lambda i: (-float(i_reaction[i]), i))
                for scale in trial_scales:
                    target_count = int(np.ceil(float(base_reactions_after) * float(scale)))
                    target_count = max(int(stage_floor_min_reactions), min(target_count, base_reactions_after))
                    if target_count >= base_reactions_after:
                        continue
                    selected_idx = sorted_keep_indices[:target_count]
                    keep_trial = np.zeros_like(keep_base, dtype=bool)
                    if selected_idx:
                        keep_trial[np.asarray(selected_idx, dtype=int)] = True
                    reactions_after_trial = int(np.sum(keep_trial))
                    if reactions_after_trial >= base_reactions_after:
                        continue
                    if reactions_after_trial < floor_min_reactions_local:
                        continue

                    t_trial_physical = pytime.perf_counter()
                    physical_trial = _evaluate_physical_gate(
                        enabled=physical_enabled,
                        nu=nu,
                        rop=rop,
                        dt=dt,
                        A=A,
                        X=X,
                        S=np.asarray(mapping.S, dtype=float),
                        keep_reactions=np.asarray(keep_trial, dtype=bool),
                        max_conservation_violation=physical_max_cons,
                        max_negative_steps=physical_max_negative_steps,
                        state_source=state_source,
                        degraded=state_degraded,
                        fallback_reason=state_fallback_reason,
                    )
                    trial_timing_physical += float(pytime.perf_counter() - t_trial_physical)
                    trial_timing_projection += float(physical_trial.get("timing_projection_s", 0.0))
                    physical_trial_passed = bool(physical_trial.get("passed", True))

                    nu_balance_trial = np.asarray(
                        np.asarray(mapping.S, dtype=float).T @ np.asarray(nu[:, keep_trial], dtype=float),
                        dtype=float,
                    )
                    species_after_trial = int(mapping.S.shape[1])
                    floor_passed_trial = bool(
                        species_after_trial >= floor_min_species_local
                        and reactions_after_trial >= floor_min_reactions_local
                    )
                    balance_metrics_trial = _compute_structural_balance_metrics(
                        nu_reduced=nu_balance_trial,
                        species_after=species_after_trial,
                        reactions_after=reactions_after_trial,
                        activity_weights=balance_activity_weights,
                        essential_cluster_mask=balance_essential_mask,
                        S_reduced=np.asarray(mapping.S, dtype=float),
                        species_before=int(n_species),
                        top_weight_mass_ratio=float(balance_bands.get("top_weight_mass_ratio", 0.80)),
                        balance_mode=str(balance_bands.get("balance_mode", "binary")),
                    )
                    balance_result_trial = _evaluate_balance_gate(balance_metrics_trial, balance_bands)
                    balance_passed_trial = bool(balance_result_trial.get("passed", True))
                    structure_passed_trial = bool(
                        floor_passed_trial
                        and balance_passed_trial
                        and (physical_trial_passed if physical_enabled else True)
                    )
                    if compression_require_structure_passed and not structure_passed_trial:
                        continue

                    trial_pass_rate_mandatory = float(base_pass_rate_mandatory)
                    trial_mean_rel_mandatory = float(base_mean_rel_mandatory)
                    trial_mean_rel_optional = float(base_mean_rel_optional)
                    quality_guard_passed = bool(
                        (base_pass_rate_mandatory - trial_pass_rate_mandatory)
                        <= (compression_guard_mandatory_pass_drop + 1.0e-12)
                    )
                    quality_guard_passed = bool(
                        quality_guard_passed
                        and (trial_mean_rel_mandatory - base_mean_rel_mandatory)
                        <= (compression_guard_mandatory_mean_delta + 1.0e-12)
                    )
                    quality_guard_passed = bool(
                        quality_guard_passed
                        and (trial_mean_rel_optional - base_mean_rel_optional)
                        <= (compression_guard_optional_mean_delta + 1.0e-12)
                    )
                    if not quality_guard_passed:
                        continue

                    candidate_key = (
                        reactions_after_trial if compression_reaction_priority else species_after_trial,
                        species_after_trial if compression_reaction_priority else reactions_after_trial,
                    )
                    if best_trial is None or candidate_key < best_trial["candidate_key"]:
                        best_trial = {
                            "candidate_key": candidate_key,
                            "keep": keep_trial,
                            "physical_result": dict(physical_trial),
                            "nu_balance": nu_balance_trial,
                            "reactions_after": reactions_after_trial,
                            "species_after": species_after_trial,
                        }

                stage_physical_elapsed_s += float(trial_timing_physical)
                stage_projection_elapsed_s += float(trial_timing_projection)
                if best_trial is not None:
                    keep = np.asarray(best_trial["keep"], dtype=bool)
                    physical_result = dict(best_trial["physical_result"])
                    nu_balance = np.asarray(best_trial["nu_balance"], dtype=float)
                    overall_selected = int(best_trial["reactions_after"])
                    overall_select_ratio = float(overall_selected / max(overall_candidates, 1))
                    compression_refine_applied = True
                    compression_refine_reaction_delta = int(base_reactions_after - int(best_trial["reactions_after"]))
                    compression_refine_species_delta = int(0)
                    prune_details = dict(prune_details)
                    prune_details["compression_refine"] = {
                        "applied": True,
                        "trials": int(compression_refine_trials),
                        "reaction_delta": int(compression_refine_reaction_delta),
                        "species_delta": int(compression_refine_species_delta),
                        "mode_effective": "baseline_grid",
                    }
                    prune_details["status"] = f"{prune_details.get('status', 'ok')}:compression_refined"

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
            gas_reactions_after = None
            surface_reactions_after = None
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
        mandatory_total_metric_count = int(getattr(eval_summary, "mandatory_total_metric_count", 0) or 0)
        valid_mandatory_metric_count = int(getattr(eval_summary, "valid_mandatory_metric_count", 0) or 0)
        invalid_mandatory_metric_count = int(getattr(eval_summary, "invalid_mandatory_metric_count", 0) or 0)
        inactive_mandatory_metric_count = int(getattr(eval_summary, "inactive_mandatory_metric_count", 0) or 0)
        active_invalid_mandatory_metric_count = int(getattr(eval_summary, "active_invalid_mandatory_metric_count", 0) or 0)
        mandatory_metric_case_pass_rates = dict(getattr(eval_summary, "mandatory_metric_case_pass_rates", {}) or {})
        mandatory_metric_valid_case_pass_min_effective = float(
            getattr(eval_summary, "mandatory_metric_valid_case_pass_min_effective", 0.0) or 0.0
        )
        mandatory_metric_validity_mode_effective = str(
            getattr(eval_summary, "mandatory_metric_validity_mode_effective", "case_pass_rate") or "case_pass_rate"
        )
        mandatory_total_gate_unit_count = int(getattr(eval_summary, "mandatory_total_gate_unit_count", 0) or 0)
        valid_mandatory_gate_unit_count = int(getattr(eval_summary, "valid_mandatory_gate_unit_count", 0) or 0)
        valid_mandatory_gate_unit_count_case_rate = int(
            getattr(
                eval_summary,
                "valid_mandatory_gate_unit_count_case_rate",
                valid_mandatory_gate_unit_count,
            )
            or 0
        )
        valid_mandatory_gate_unit_count_coverage = int(
            getattr(
                eval_summary,
                "valid_mandatory_gate_unit_count_coverage",
                valid_mandatory_gate_unit_count,
            )
            or 0
        )
        mandatory_validity_basis_effective = str(
            getattr(eval_summary, "mandatory_validity_basis_effective", "coverage_evaluable")
            or "coverage_evaluable"
        )
        mandatory_quality_gate_unit_count = int(
            getattr(eval_summary, "mandatory_quality_gate_unit_count", mandatory_total_gate_unit_count)
            or mandatory_total_gate_unit_count
        )
        mandatory_quality_metric_count = int(
            getattr(eval_summary, "mandatory_quality_metric_count", mandatory_total_metric_count)
            or mandatory_total_metric_count
        )
        mandatory_gate_unit_case_pass_rates = dict(
            getattr(eval_summary, "mandatory_gate_unit_case_pass_rates", {}) or {}
        )
        mandatory_gate_unit_evaluable_case_rates = dict(
            getattr(eval_summary, "mandatory_gate_unit_evaluable_case_rates", {}) or {}
        )
        mandatory_gate_unit_valid_count_shadow_evaluable_ratio = int(
            getattr(eval_summary, "mandatory_gate_unit_valid_count_shadow_evaluable_ratio", 0) or 0
        )
        mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective = float(
            getattr(eval_summary, "mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective", 0.25)
            or 0.25
        )
        active_invalid_mandatory_gate_unit_keys = [
            str(x) for x in list(getattr(eval_summary, "active_invalid_mandatory_gate_unit_keys", []) or [])
        ]
        mandatory_gate_unit_mode_effective = str(
            getattr(eval_summary, "mandatory_gate_unit_mode_effective", "species_family_quorum")
            or "species_family_quorum"
        )
        mandatory_species_family_score_mode_effective = str(
            getattr(eval_summary, "mandatory_species_family_score_mode_effective", "uniform")
            or "uniform"
        )
        mandatory_quality_scope_effective = str(
            getattr(eval_summary, "mandatory_quality_scope_effective", "valid_only") or "valid_only"
        )
        mandatory_tail_scope_effective = str(
            getattr(eval_summary, "mandatory_tail_scope_effective", "quality_scope") or "quality_scope"
        )
        mandatory_species_family_case_pass_min_effective = float(
            getattr(eval_summary, "mandatory_species_family_case_pass_min_effective", 0.67) or 0.67
        )
        min_valid_mandatory_count_effective = int(getattr(eval_summary, "min_valid_mandatory_count_effective", 0) or 0)
        mandatory_validity_passed = bool(getattr(eval_summary, "mandatory_validity_passed", True))
        pass_rate_mandatory_case = float(getattr(eval_summary, "pass_rate_mandatory_case", eval_summary.pass_rate) or 0.0)
        pass_rate_mandatory_case_all_units = float(
            getattr(eval_summary, "pass_rate_mandatory_case_all_units", pass_rate_mandatory_case) or 0.0
        )
        pass_rate_mandatory_case_all_required = float(
            getattr(eval_summary, "pass_rate_mandatory_case_all_required", pass_rate_mandatory_case) or 0.0
        )
        pass_rate_mandatory_case_ratio_mean = float(
            getattr(eval_summary, "pass_rate_mandatory_case_ratio_mean", pass_rate_mandatory_case) or 0.0
        )
        pass_rate_mandatory_case_all_required_all_units = float(
            getattr(
                eval_summary,
                "pass_rate_mandatory_case_all_required_all_units",
                pass_rate_mandatory_case_all_required,
            )
            or 0.0
        )
        pass_rate_mandatory_case_ratio_mean_all_units = float(
            getattr(
                eval_summary,
                "pass_rate_mandatory_case_ratio_mean_all_units",
                pass_rate_mandatory_case_ratio_mean,
            )
            or 0.0
        )
        mandatory_case_mode_effective = str(
            getattr(eval_summary, "mandatory_case_mode_effective", "ratio_mean") or "ratio_mean"
        )
        mandatory_case_unit_weight_mode_effective = str(
            getattr(eval_summary, "mandatory_case_unit_weight_mode_effective", "uniform") or "uniform"
        )
        pass_rate_optional_case = float(getattr(eval_summary, "pass_rate_optional_case", eval_summary.pass_rate) or 0.0)
        pass_rate_optional_metric_mean = float(
            getattr(eval_summary, "pass_rate_optional_metric_mean", eval_summary.pass_rate) or 0.0
        )
        pass_rate_all_metric_legacy = float(
            getattr(eval_summary, "pass_rate_all_metric_legacy", eval_summary.pass_rate) or 0.0
        )
        mean_rel_diff_mandatory = float(
            getattr(eval_summary, "mean_rel_diff_mandatory", eval_summary.mean_rel_diff or 0.0) or 0.0
        )
        mean_rel_diff_mandatory_all_units = float(
            getattr(eval_summary, "mean_rel_diff_mandatory_all_units", mean_rel_diff_mandatory)
            or mean_rel_diff_mandatory
        )
        mean_rel_diff_mandatory_raw = float(
            getattr(eval_summary, "mean_rel_diff_mandatory_raw", mean_rel_diff_mandatory) or 0.0
        )
        mean_rel_diff_mandatory_family_weighted = float(
            getattr(eval_summary, "mean_rel_diff_mandatory_family_weighted", mean_rel_diff_mandatory) or 0.0
        )
        mean_rel_diff_mandatory_winsorized = float(
            getattr(eval_summary, "mean_rel_diff_mandatory_winsorized", mean_rel_diff_mandatory) or 0.0
        )
        mandatory_rel_outlier_ratio = float(
            getattr(eval_summary, "mandatory_rel_outlier_ratio", 0.0) or 0.0
        )
        mandatory_rel_outlier_ratio_all_units = float(
            getattr(eval_summary, "mandatory_rel_outlier_ratio_all_units", mandatory_rel_outlier_ratio)
            or mandatory_rel_outlier_ratio
        )
        mandatory_rel_outlier_ratio_max_effective = float(
            getattr(eval_summary, "mandatory_rel_outlier_ratio_max_effective", 0.20) or 0.20
        )
        mandatory_rel_diff_p95 = float(
            getattr(eval_summary, "mandatory_rel_diff_p95", mean_rel_diff_mandatory_raw) or 0.0
        )
        mandatory_rel_diff_p95_all_units = float(
            getattr(eval_summary, "mandatory_rel_diff_p95_all_units", mandatory_rel_diff_p95)
            or mandatory_rel_diff_p95
        )
        mandatory_tail_guard_passed = bool(getattr(eval_summary, "mandatory_tail_guard_passed", True))
        mandatory_tail_guard_mode_effective = str(
            getattr(eval_summary, "mandatory_tail_guard_mode_effective", "p95") or "p95"
        )
        mandatory_tail_rel_diff_max_effective = float(
            getattr(eval_summary, "mandatory_tail_rel_diff_max_effective", 1.50) or 1.50
        )
        mandatory_quality_scope_empty = bool(
            getattr(eval_summary, "mandatory_quality_scope_empty", False)
        )
        mandatory_mean_aggregation_effective = str(
            getattr(eval_summary, "mandatory_mean_aggregation_effective", "raw") or "raw"
        )
        mandatory_mean_mode_effective = str(
            getattr(eval_summary, "mandatory_mean_mode_effective", "winsorized") or "winsorized"
        )
        mean_rel_diff_optional = float(
            getattr(eval_summary, "mean_rel_diff_optional", eval_summary.mean_rel_diff or 0.0) or 0.0
        )
        mean_rel_diff_all_metric_legacy = float(
            getattr(eval_summary, "mean_rel_diff_all_metric_legacy", eval_summary.mean_rel_diff or 0.0) or 0.0
        )
        error_gate_score = float(getattr(eval_summary, "error_gate_score", pass_rate_mandatory_case) or 0.0)
        effective_metric_count = int(
            getattr(eval_summary, "effective_metric_count", getattr(eval_summary, "qoi_metrics_count", 0)) or 0
        )
        suppressed_low_signal_metric_count = int(
            getattr(eval_summary, "suppressed_low_signal_metric_count", 0) or 0
        )
        error_gate_passed = bool(
            getattr(
                eval_summary,
                "error_gate_passed",
                (eval_summary.pass_rate >= min_pass_rate and (eval_summary.mean_rel_diff or 0.0) <= max_mean_rel),
            )
        )
        coverage_gate_passed = bool(getattr(eval_summary, "coverage_gate_passed", mandatory_validity_passed))
        mandatory_quality_passed = bool(
            getattr(eval_summary, "mandatory_quality_passed", getattr(eval_summary, "mandatory_error_passed", True))
        )
        optional_quality_passed = bool(
            getattr(eval_summary, "optional_quality_passed", getattr(eval_summary, "optional_error_passed", True))
        )
        mandatory_error_passed = bool(
            getattr(eval_summary, "mandatory_error_passed", mandatory_quality_passed)
        )
        optional_error_passed = bool(
            getattr(eval_summary, "optional_error_passed", optional_quality_passed)
        )
        mandatory_tail_guard_triggered = bool(
            getattr(eval_summary, "mandatory_tail_guard_triggered", False)
        )
        mandatory_tail_guard_hard_applied = bool(
            getattr(eval_summary, "mandatory_tail_guard_hard_applied", False)
        )
        mandatory_tail_guard_policy_effective = str(
            getattr(eval_summary, "mandatory_tail_guard_policy_effective", "conditional_hard")
            or "conditional_hard"
        )
        mandatory_tail_activation_ratio_min_effective = float(
            getattr(eval_summary, "mandatory_tail_activation_ratio_min_effective", 0.10) or 0.10
        )
        mandatory_tail_exceed_ref_effective = str(
            getattr(eval_summary, "mandatory_tail_exceed_ref_effective", "tail_max")
            or "tail_max"
        )
        mandatory_tail_exceed_ratio = float(
            getattr(eval_summary, "mandatory_tail_exceed_ratio", 0.0) or 0.0
        )
        mandatory_error_include_validity_effective = bool(
            getattr(eval_summary, "mandatory_error_include_validity_effective", False)
        )
        error_fail_reason_primary = str(
            getattr(eval_summary, "error_fail_reason_primary", "none") or "none"
        )
        evaluation_contract_version = str(
            getattr(eval_summary, "evaluation_contract_version", contract_cfg.get("version", "v1"))
            or contract_cfg.get("version", "v1")
        )
        metric_taxonomy_profile_effective = str(
            getattr(
                eval_summary,
                "metric_taxonomy_profile_effective",
                metric_taxonomy_resolved.get("profile", "legacy_builtin"),
            )
            or metric_taxonomy_resolved.get("profile", "legacy_builtin")
        )
        diagnostic_schema_ok = bool(getattr(eval_summary, "diagnostic_schema_ok", True))

        replay_validity_cfg = dict(eval_cfg.get("gate_metric_validity") or {})
        replay_trust_detail = _evaluate_metric_replay_health_trust(
            metric_clip_ratio=float(getattr(eval_summary, "metric_clip_ratio", 0.0) or 0.0),
            guardrail_trigger_ratio=float(getattr(eval_summary, "metric_clip_guardrail_trigger_ratio", 0.0) or 0.0),
            max_metric_clip_ratio=float(replay_validity_cfg.get("max_metric_clip_ratio", 1.0)),
            min_guardrail_trigger_ratio=float(replay_validity_cfg.get("min_guardrail_trigger_ratio", 0.02)),
        )
        stage_metric_clip_guardrail_trigger_ratio = float(replay_trust_detail.get("metric_clip_guardrail_trigger_ratio", 0.0))
        stage_replay_health_trust_invalid = bool(replay_trust_detail.get("replay_health_trust_invalid", False))

        species_deficit = max(float(floor_min_species - species_after), 0.0)
        reaction_deficit = max(float(floor_min_reactions - reactions_after), 0.0)
        species_deficit_ratio = species_deficit / float(max(1, floor_min_species))
        reaction_deficit_ratio = reaction_deficit / float(max(1, floor_min_reactions))
        domain_deficit = max(species_deficit_ratio, reaction_deficit_ratio)

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
            "structure_feedback_multiplier": float(structure_feedback_multiplier),
            "pooling_hard_ban_violations": int(stage_pooling_metrics.get("hard_ban_violations", 0)),
            "pooling_constraint_loss": float((stage_pooling_metrics.get("train_metrics") or {}).get("constraint_loss", 0.0)),
            "pooling_clusters": int((stage_pooling_metrics.get("train_metrics") or {}).get("n_clusters", mapping.S.shape[1])),
            "pooling_graph_kind": str(stage_pooling_metrics.get("graph_kind", "")),
            "pooling_model_type": str((stage_pooling_metrics.get("model_info") or {}).get("model_type", "")),
            "pooling_candidate_count": int(stage_pooling_metrics.get("candidate_count", 0)),
            "pooling_candidate_unique_count": int(stage_pooling_metrics.get("candidate_unique_count", 0)),
            "pooling_candidate_selected_backend": str(stage_pooling_metrics.get("candidate_selected_backend", "")),
            "pooling_candidate_selected_source": str(stage_pooling_metrics.get("candidate_selected_source", "backend")),
            "pooling_candidate_selected_coverage_proxy": float(
                stage_pooling_metrics.get(
                    "candidate_selected_coverage_proxy",
                    (stage_pooling_metrics.get("train_metrics") or {}).get("coverage_proxy", 0.0),
                )
            ),
            "pooling_candidate_selected_dynamics_recon_error": float(
                stage_pooling_metrics.get("candidate_selected_dynamics_recon_error", 0.0)
            ),
            "pooling_candidate_selected_max_cluster_size_ratio": float(
                stage_pooling_metrics.get(
                    "candidate_selected_max_cluster_size_ratio",
                    (stage_pooling_metrics.get("train_metrics") or {}).get("max_cluster_size_ratio", 0.0),
                )
            ),
            "pooling_candidate_scores": list(stage_pooling_metrics.get("candidate_scores") or []),
            "pooling_bridge_enabled": bool(pooling_bridge_enabled if reduction_mode == "pooling" else True),
            "pooling_bridge_mode": stage_pooling_bridge_mode,
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
            "structure_deficit_score": float(
                _structure_deficit_score(
                    {
                        "balance_margin": balance_margin,
                        "floor_passed": floor_passed,
                        "cluster_guard_passed": cluster_guard_passed,
                        "physical_gate_passed": bool(physical_result["passed"]),
                    }
                )
            ),
            "mandatory_total_metric_count": int(mandatory_total_metric_count),
            "valid_mandatory_metric_count": int(valid_mandatory_metric_count),
            "invalid_mandatory_metric_count": int(invalid_mandatory_metric_count),
            "inactive_mandatory_metric_count": int(inactive_mandatory_metric_count),
            "active_invalid_mandatory_metric_count": int(active_invalid_mandatory_metric_count),
            "mandatory_metric_case_pass_rates": {str(k): float(v) for k, v in mandatory_metric_case_pass_rates.items()},
            "mandatory_metric_valid_case_pass_min_effective": float(mandatory_metric_valid_case_pass_min_effective),
            "mandatory_metric_validity_mode_effective": str(mandatory_metric_validity_mode_effective),
            "mandatory_total_gate_unit_count": int(mandatory_total_gate_unit_count),
            "valid_mandatory_gate_unit_count": int(valid_mandatory_gate_unit_count),
            "valid_mandatory_gate_unit_count_case_rate": int(valid_mandatory_gate_unit_count_case_rate),
            "valid_mandatory_gate_unit_count_coverage": int(valid_mandatory_gate_unit_count_coverage),
            "mandatory_gate_unit_valid_count_shadow_evaluable_ratio": int(
                mandatory_gate_unit_valid_count_shadow_evaluable_ratio
            ),
            "mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective": float(
                mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective
            ),
            "mandatory_validity_basis_effective": str(mandatory_validity_basis_effective),
            "mandatory_quality_gate_unit_count": int(mandatory_quality_gate_unit_count),
            "mandatory_quality_metric_count": int(mandatory_quality_metric_count),
            "active_invalid_mandatory_gate_unit_keys": list(active_invalid_mandatory_gate_unit_keys),
            "mandatory_gate_unit_case_pass_rates": {
                str(k): float(v) for k, v in mandatory_gate_unit_case_pass_rates.items()
            },
            "mandatory_gate_unit_evaluable_case_rates": {
                str(k): float(v) for k, v in mandatory_gate_unit_evaluable_case_rates.items()
            },
            "mandatory_gate_unit_mode_effective": str(mandatory_gate_unit_mode_effective),
            "mandatory_species_family_score_mode_effective": str(
                mandatory_species_family_score_mode_effective
            ),
            "mandatory_quality_scope_effective": str(mandatory_quality_scope_effective),
            "mandatory_tail_scope_effective": str(mandatory_tail_scope_effective),
            "mandatory_species_family_case_pass_min_effective": float(
                mandatory_species_family_case_pass_min_effective
            ),
            "min_valid_mandatory_count_effective": int(min_valid_mandatory_count_effective),
            "mandatory_validity_passed": bool(mandatory_validity_passed),
            "pass_rate_mandatory_case": float(pass_rate_mandatory_case),
            "pass_rate_mandatory_case_all_units": float(pass_rate_mandatory_case_all_units),
            "pass_rate_mandatory_case_all_required": float(pass_rate_mandatory_case_all_required),
            "pass_rate_mandatory_case_ratio_mean": float(pass_rate_mandatory_case_ratio_mean),
            "pass_rate_mandatory_case_all_required_all_units": float(
                pass_rate_mandatory_case_all_required_all_units
            ),
            "pass_rate_mandatory_case_ratio_mean_all_units": float(
                pass_rate_mandatory_case_ratio_mean_all_units
            ),
            "pass_rate_all_metric_legacy": float(pass_rate_all_metric_legacy),
            "mandatory_case_mode_effective": str(mandatory_case_mode_effective),
            "mandatory_case_unit_weight_mode_effective": str(
                mandatory_case_unit_weight_mode_effective
            ),
            "pass_rate_optional_case": float(pass_rate_optional_case),
            "pass_rate_optional_metric_mean": float(pass_rate_optional_metric_mean),
            "mean_rel_diff_mandatory": float(mean_rel_diff_mandatory),
            "mean_rel_diff_mandatory_all_units": float(mean_rel_diff_mandatory_all_units),
            "mean_rel_diff_mandatory_raw": float(mean_rel_diff_mandatory_raw),
            "mean_rel_diff_mandatory_family_weighted": float(mean_rel_diff_mandatory_family_weighted),
            "mean_rel_diff_mandatory_winsorized": float(mean_rel_diff_mandatory_winsorized),
            "mandatory_rel_outlier_ratio": float(mandatory_rel_outlier_ratio),
            "mandatory_rel_outlier_ratio_all_units": float(mandatory_rel_outlier_ratio_all_units),
            "mandatory_rel_outlier_ratio_max_effective": float(mandatory_rel_outlier_ratio_max_effective),
            "mandatory_rel_diff_p95": float(mandatory_rel_diff_p95),
            "mandatory_rel_diff_p95_all_units": float(mandatory_rel_diff_p95_all_units),
            "mandatory_tail_guard_passed": bool(mandatory_tail_guard_passed),
            "mandatory_tail_guard_triggered": bool(mandatory_tail_guard_triggered),
            "mandatory_tail_guard_hard_applied": bool(mandatory_tail_guard_hard_applied),
            "mandatory_tail_guard_mode_effective": str(mandatory_tail_guard_mode_effective),
            "mandatory_tail_guard_policy_effective": str(mandatory_tail_guard_policy_effective),
            "mandatory_tail_activation_ratio_min_effective": float(
                mandatory_tail_activation_ratio_min_effective
            ),
            "mandatory_tail_exceed_ref_effective": str(mandatory_tail_exceed_ref_effective),
            "mandatory_tail_exceed_ratio": float(mandatory_tail_exceed_ratio),
            "mandatory_tail_rel_diff_max_effective": float(mandatory_tail_rel_diff_max_effective),
            "mandatory_quality_scope_empty": bool(mandatory_quality_scope_empty),
            "mandatory_mean_aggregation_effective": str(mandatory_mean_aggregation_effective),
            "mandatory_mean_mode_effective": str(mandatory_mean_mode_effective),
            "mean_rel_diff_optional": float(mean_rel_diff_optional),
            "mean_rel_diff_all_metric_legacy": float(mean_rel_diff_all_metric_legacy),
            "error_gate_score": float(error_gate_score),
            "error_gate_passed": bool(error_gate_passed),
            "coverage_gate_passed": bool(coverage_gate_passed),
            "mandatory_quality_passed": bool(mandatory_quality_passed),
            "optional_quality_passed": bool(optional_quality_passed),
            "mandatory_error_passed": bool(mandatory_error_passed),
            "optional_error_passed": bool(optional_error_passed),
            "mandatory_error_include_validity_effective": bool(
                mandatory_error_include_validity_effective
            ),
            "error_fail_reason_primary": str(error_fail_reason_primary),
            "evaluation_contract_version": str(evaluation_contract_version),
            "metric_taxonomy_profile_effective": str(metric_taxonomy_profile_effective),
            "diagnostic_schema_ok": bool(diagnostic_schema_ok),
            "effective_metric_count": int(effective_metric_count),
            "suppressed_low_signal_metric_count": int(suppressed_low_signal_metric_count),
            "metric_clip_guardrail_trigger_ratio": float(stage_metric_clip_guardrail_trigger_ratio),
            "replay_health_trust_invalid": bool(stage_replay_health_trust_invalid),
            "metric_drift_raw": float(stage_metric_drift_raw),
            "metric_drift_effective": float(stage_metric_drift),
            "metric_drift_effective_cap": float(surrogate_drift_effective_cap_for_eval),
            "compression_refine_applied": bool(compression_refine_applied),
            "compression_refine_trials": int(compression_refine_trials),
            "compression_refine_reaction_delta": int(compression_refine_reaction_delta),
            "compression_refine_species_delta": int(compression_refine_species_delta),
            "compression_refine_mode_effective": str(compression_refine_mode_effective),
            "compression_refine_guard_passed": bool(compression_refine_guard_passed),
            "timing_stage_s": float(0.0),
            "timing_pooling_fit_s": float(stage_pooling_fit_elapsed_s),
            "timing_bridge_s": float(stage_bridge_elapsed_s),
            "timing_surrogate_eval_s": float(stage_surrogate_elapsed_s),
            "timing_physical_gate_s": float(stage_physical_elapsed_s),
            "timing_projection_s": float(stage_projection_elapsed_s),
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
        gate_passed = bool(
            coverage_gate_passed
            and error_gate_passed
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
                "structure_deficit_score": float(
                    _structure_deficit_score(
                        {
                            "balance_margin": balance_margin,
                            "floor_passed": floor_passed,
                            "cluster_guard_passed": cluster_guard_passed,
                            "physical_gate_passed": physical_pass,
                        }
                    )
                ),
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
                "mandatory_total_metric_count": int(mandatory_total_metric_count),
                "valid_mandatory_metric_count": int(valid_mandatory_metric_count),
                "invalid_mandatory_metric_count": int(invalid_mandatory_metric_count),
                "inactive_mandatory_metric_count": int(inactive_mandatory_metric_count),
                "active_invalid_mandatory_metric_count": int(active_invalid_mandatory_metric_count),
                "mandatory_metric_case_pass_rates": {
                    str(k): float(v) for k, v in mandatory_metric_case_pass_rates.items()
                },
                "mandatory_metric_valid_case_pass_min_effective": float(mandatory_metric_valid_case_pass_min_effective),
                "mandatory_metric_validity_mode_effective": str(mandatory_metric_validity_mode_effective),
                "mandatory_total_gate_unit_count": int(mandatory_total_gate_unit_count),
                "valid_mandatory_gate_unit_count": int(valid_mandatory_gate_unit_count),
                "valid_mandatory_gate_unit_count_case_rate": int(valid_mandatory_gate_unit_count_case_rate),
                "valid_mandatory_gate_unit_count_coverage": int(valid_mandatory_gate_unit_count_coverage),
                "mandatory_gate_unit_valid_count_shadow_evaluable_ratio": int(
                    mandatory_gate_unit_valid_count_shadow_evaluable_ratio
                ),
                "mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective": float(
                    mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective
                ),
                "mandatory_validity_basis_effective": str(mandatory_validity_basis_effective),
                "mandatory_quality_gate_unit_count": int(mandatory_quality_gate_unit_count),
                "mandatory_quality_metric_count": int(mandatory_quality_metric_count),
                "active_invalid_mandatory_gate_unit_keys": list(active_invalid_mandatory_gate_unit_keys),
                "mandatory_gate_unit_case_pass_rates": {
                    str(k): float(v) for k, v in mandatory_gate_unit_case_pass_rates.items()
                },
                "mandatory_gate_unit_evaluable_case_rates": {
                    str(k): float(v) for k, v in mandatory_gate_unit_evaluable_case_rates.items()
                },
                "mandatory_gate_unit_mode_effective": str(mandatory_gate_unit_mode_effective),
                "mandatory_species_family_score_mode_effective": str(
                    mandatory_species_family_score_mode_effective
                ),
                "mandatory_quality_scope_effective": str(mandatory_quality_scope_effective),
                "mandatory_tail_scope_effective": str(mandatory_tail_scope_effective),
                "mandatory_species_family_case_pass_min_effective": float(
                    mandatory_species_family_case_pass_min_effective
                ),
                "min_valid_mandatory_count_effective": int(min_valid_mandatory_count_effective),
                "mandatory_validity_passed": bool(mandatory_validity_passed),
                "pass_rate_mandatory_case": float(pass_rate_mandatory_case),
                "pass_rate_mandatory_case_all_units": float(pass_rate_mandatory_case_all_units),
                "pass_rate_mandatory_case_all_required": float(pass_rate_mandatory_case_all_required),
                "pass_rate_mandatory_case_ratio_mean": float(pass_rate_mandatory_case_ratio_mean),
                "pass_rate_mandatory_case_all_required_all_units": float(
                    pass_rate_mandatory_case_all_required_all_units
                ),
                "pass_rate_mandatory_case_ratio_mean_all_units": float(
                    pass_rate_mandatory_case_ratio_mean_all_units
                ),
                "pass_rate_all_metric_legacy": float(pass_rate_all_metric_legacy),
                "mandatory_case_mode_effective": str(mandatory_case_mode_effective),
                "mandatory_case_unit_weight_mode_effective": str(
                    mandatory_case_unit_weight_mode_effective
                ),
                "pass_rate_optional_case": float(pass_rate_optional_case),
                "pass_rate_optional_metric_mean": float(pass_rate_optional_metric_mean),
                "mean_rel_diff_mandatory": float(mean_rel_diff_mandatory),
                "mean_rel_diff_mandatory_all_units": float(mean_rel_diff_mandatory_all_units),
                "mean_rel_diff_mandatory_raw": float(mean_rel_diff_mandatory_raw),
                "mean_rel_diff_mandatory_family_weighted": float(mean_rel_diff_mandatory_family_weighted),
                "mean_rel_diff_mandatory_winsorized": float(mean_rel_diff_mandatory_winsorized),
                "mandatory_rel_outlier_ratio": float(mandatory_rel_outlier_ratio),
                "mandatory_rel_outlier_ratio_all_units": float(mandatory_rel_outlier_ratio_all_units),
                "mandatory_rel_outlier_ratio_max_effective": float(mandatory_rel_outlier_ratio_max_effective),
                "mandatory_rel_diff_p95": float(mandatory_rel_diff_p95),
                "mandatory_rel_diff_p95_all_units": float(mandatory_rel_diff_p95_all_units),
                "mandatory_tail_guard_passed": bool(mandatory_tail_guard_passed),
                "mandatory_tail_guard_triggered": bool(mandatory_tail_guard_triggered),
                "mandatory_tail_guard_hard_applied": bool(mandatory_tail_guard_hard_applied),
                "mandatory_tail_guard_mode_effective": str(mandatory_tail_guard_mode_effective),
                "mandatory_tail_guard_policy_effective": str(mandatory_tail_guard_policy_effective),
                "mandatory_tail_activation_ratio_min_effective": float(
                    mandatory_tail_activation_ratio_min_effective
                ),
                "mandatory_tail_exceed_ref_effective": str(mandatory_tail_exceed_ref_effective),
                "mandatory_tail_exceed_ratio": float(mandatory_tail_exceed_ratio),
                "mandatory_tail_rel_diff_max_effective": float(mandatory_tail_rel_diff_max_effective),
                "mandatory_quality_scope_empty": bool(mandatory_quality_scope_empty),
                "mandatory_mean_aggregation_effective": str(mandatory_mean_aggregation_effective),
                "mandatory_mean_mode_effective": str(mandatory_mean_mode_effective),
                "mean_rel_diff_optional": float(mean_rel_diff_optional),
                "mean_rel_diff_all_metric_legacy": float(mean_rel_diff_all_metric_legacy),
                "error_gate_score": float(error_gate_score),
                "error_gate_passed": bool(error_gate_passed),
                "coverage_gate_passed": bool(coverage_gate_passed),
                "mandatory_quality_passed": bool(mandatory_quality_passed),
                "optional_quality_passed": bool(optional_quality_passed),
                "mandatory_error_passed": bool(mandatory_error_passed),
                "optional_error_passed": bool(optional_error_passed),
                "mandatory_error_include_validity_effective": bool(
                    mandatory_error_include_validity_effective
                ),
                "error_fail_reason_primary": str(error_fail_reason_primary),
                "evaluation_contract_version": str(evaluation_contract_version),
                "metric_taxonomy_profile_effective": str(metric_taxonomy_profile_effective),
                "diagnostic_schema_ok": bool(diagnostic_schema_ok),
                "effective_metric_count": int(effective_metric_count),
                "suppressed_low_signal_metric_count": int(suppressed_low_signal_metric_count),
                "metric_clip_guardrail_trigger_ratio": float(stage_metric_clip_guardrail_trigger_ratio),
                "replay_health_trust_invalid": bool(stage_replay_health_trust_invalid),
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
                "metric_drift_raw": float(stage_metric_drift_raw),
                "metric_drift_effective": float(stage_metric_drift),
                "metric_drift_effective_cap": float(surrogate_drift_effective_cap_for_eval),
                "compression_refine_applied": bool(compression_refine_applied),
                "compression_refine_trials": int(compression_refine_trials),
                "compression_refine_reaction_delta": int(compression_refine_reaction_delta),
                "compression_refine_species_delta": int(compression_refine_species_delta),
                "compression_refine_mode_effective": str(compression_refine_mode_effective),
                "compression_refine_guard_passed": bool(compression_refine_guard_passed),
                "learnckpp_fallback_reason": learnckpp_fallback_reason,
                "learnckpp_target_keep_ratio": float(learnckpp_target_keep_ratio),
                "structure_feedback_multiplier": float(structure_feedback_multiplier),
                "pooling_clusters": int((stage_pooling_metrics.get("train_metrics") or {}).get("n_clusters", mapping.S.shape[1])),
                "pooling_constraint_loss": float((stage_pooling_metrics.get("train_metrics") or {}).get("constraint_loss", 0.0)),
                "pooling_hard_ban_violations": int(stage_pooling_metrics.get("hard_ban_violations", 0)),
                "pooling_candidate_count": int(stage_pooling_metrics.get("candidate_count", 0)),
                "pooling_candidate_unique_count": int(stage_pooling_metrics.get("candidate_unique_count", 0)),
                "pooling_candidate_selected_backend": str(stage_pooling_metrics.get("candidate_selected_backend", "")),
                "pooling_candidate_selected_source": str(stage_pooling_metrics.get("candidate_selected_source", "backend")),
                "pooling_candidate_selected_coverage_proxy": float(
                    stage_pooling_metrics.get(
                        "candidate_selected_coverage_proxy",
                        (stage_pooling_metrics.get("train_metrics") or {}).get("coverage_proxy", 0.0),
                    )
                ),
                "pooling_candidate_selected_dynamics_recon_error": float(
                    stage_pooling_metrics.get("candidate_selected_dynamics_recon_error", 0.0)
                ),
                "pooling_candidate_selected_max_cluster_size_ratio": float(
                    stage_pooling_metrics.get(
                        "candidate_selected_max_cluster_size_ratio",
                        (stage_pooling_metrics.get("train_metrics") or {}).get("max_cluster_size_ratio", 0.0),
                    )
                ),
                "pooling_candidate_scores": list(stage_pooling_metrics.get("candidate_scores") or []),
                "pooling_bridge_enabled": bool(pooling_bridge_enabled if reduction_mode == "pooling" else True),
                "pooling_bridge_mode": stage_pooling_bridge_mode,
                "pooling_artifact_path": stage_pooling_artifact_path,
                "timing_stage_s": float(0.0),
                "timing_pooling_fit_s": float(stage_pooling_fit_elapsed_s),
                "timing_bridge_s": float(stage_bridge_elapsed_s),
                "timing_surrogate_eval_s": float(stage_surrogate_elapsed_s),
                "timing_physical_gate_s": float(stage_physical_elapsed_s),
                "timing_projection_s": float(stage_projection_elapsed_s),
                "_floors": floors,
                "_selection_max_mean_rel": max_mean_rel,
                "_selection_use_raw_drift": bool(surrogate_drift_selection_use_raw),
                "_selection_raw_drift_cap": float(surrogate_drift_raw_cap_for_selection),
            }
        )
        stage_elapsed_s = float(pytime.perf_counter() - stage_started)
        timing_stage_s[name] = stage_elapsed_s
        timing_pooling_fit_s[name] = float(stage_pooling_fit_elapsed_s)
        timing_bridge_s[name] = float(stage_bridge_elapsed_s)
        timing_surrogate_eval_s[name] = float(stage_surrogate_elapsed_s)
        timing_physical_gate_s[name] = float(stage_physical_elapsed_s)
        timing_projection_s[name] = float(stage_projection_elapsed_s)
        stage_rows[-1]["timing_stage_s"] = stage_elapsed_s
        gate_evidence_by_stage[name]["timing_stage_s"] = stage_elapsed_s
        runtime_guard.mark_progress(f"stage_{name}_completed")
        prev_stage_mean_rel = float(eval_summary.mean_rel_diff or 0.0)
        prev_stage_physical = dict(physical_result)
        prev_stage_structure = {
            "domain_deficit": float(domain_deficit),
            "balance_margin": float(balance_margin),
            "floor_passed": bool(floor_passed),
            "balance_passed": bool(balance_passed),
        }

    selection_result = _select_stage_physics_first(stage_rows, cfg)
    selected = dict(selection_result["selected"])
    for key in ("_floors", "_selection_max_mean_rel", "_selection_use_raw_drift", "_selection_raw_drift_cap"):
        selected.pop(key, None)
    for row in stage_rows:
        row.pop("_floors", None)
        row.pop("_selection_max_mean_rel", None)
        row.pop("_selection_use_raw_drift", None)
        row.pop("_selection_raw_drift_cap", None)
        row_blockers = _derive_blockers(row)
        row.update(row_blockers)
        stage_name = str(row.get("stage") or "")
        if stage_name:
            gate_evidence_by_stage.setdefault(stage_name, {}).update(row_blockers)
    report_dir = Path(cfg.get("report_dir", "reports")) / args.run_id
    selected_stage_name = str(selected["stage"])
    selected_gate_evidence = dict(gate_evidence_by_stage.get(selected_stage_name) or {})
    selected_blockers = _derive_blockers(selected)
    selected.update(selected_blockers)
    selected["selection_pool_kind"] = str(selected.get("selection_pool_kind", "all"))
    selected["structure_deficit_score"] = float(selected.get("structure_deficit_score", 0.0))
    selected["metric_drift_raw"] = float(selected.get("metric_drift_raw", selected.get("metric_drift_effective", 1.0)))
    selected["metric_drift_effective_cap"] = float(
        selected.get("metric_drift_effective_cap", surrogate_drift_effective_cap_for_eval)
    )
    selected["selection_quality_score_raw_drift"] = float(
        selected.get("selection_quality_score_raw_drift", _selection_quality_score_raw_drift(selected))
    )
    selected["compression_refine_applied"] = bool(selected.get("compression_refine_applied", False))
    selected["compression_refine_trials"] = int(selected.get("compression_refine_trials", 0) or 0)
    selected["compression_refine_reaction_delta"] = int(selected.get("compression_refine_reaction_delta", 0) or 0)
    selected["compression_refine_species_delta"] = int(selected.get("compression_refine_species_delta", 0) or 0)
    selected["compression_refine_mode_effective"] = str(selected.get("compression_refine_mode_effective", "none"))
    selected["compression_refine_guard_passed"] = bool(selected.get("compression_refine_guard_passed", True))
    selected_gate_evidence["selection_pool_kind"] = str(selected.get("selection_pool_kind", "all"))
    selected_gate_evidence["structure_deficit_score"] = float(selected.get("structure_deficit_score", 0.0))
    selected_gate_evidence["metric_drift_raw"] = float(selected.get("metric_drift_raw", 1.0))
    selected_gate_evidence["metric_drift_effective_cap"] = float(
        selected.get("metric_drift_effective_cap", surrogate_drift_effective_cap_for_eval)
    )
    selected_gate_evidence["selection_quality_score_raw_drift"] = float(
        selected.get("selection_quality_score_raw_drift", 0.0)
    )
    selected_gate_evidence["compression_refine_applied"] = bool(selected.get("compression_refine_applied", False))
    selected_gate_evidence["compression_refine_trials"] = int(selected.get("compression_refine_trials", 0))
    selected_gate_evidence["compression_refine_reaction_delta"] = int(
        selected.get("compression_refine_reaction_delta", 0)
    )
    selected_gate_evidence["compression_refine_species_delta"] = int(
        selected.get("compression_refine_species_delta", 0)
    )
    selected_gate_evidence["compression_refine_mode_effective"] = str(
        selected.get("compression_refine_mode_effective", "none")
    )
    selected_gate_evidence["compression_refine_guard_passed"] = bool(
        selected.get("compression_refine_guard_passed", True)
    )
    selected_gate_evidence["evaluation_contract_version"] = str(
        selected.get("evaluation_contract_version", contract_cfg.get("version", "v1"))
    )
    selected_gate_evidence["metric_taxonomy_profile_effective"] = str(
        selected.get("metric_taxonomy_profile_effective", metric_taxonomy_resolved.get("profile", "legacy_builtin"))
    )
    selected_gate_evidence["diagnostic_schema_ok"] = bool(selected.get("diagnostic_schema_ok", True))
    gate_evidence_by_stage.setdefault(selected_stage_name, {})["selection_pool_kind"] = str(
        selected.get("selection_pool_kind", "all")
    )
    gate_evidence_by_stage.setdefault(selected_stage_name, {})["structure_deficit_score"] = float(
        selected.get("structure_deficit_score", 0.0)
    )
    gate_evidence_by_stage.setdefault(selected_stage_name, {})["metric_drift_raw"] = float(
        selected.get("metric_drift_raw", 1.0)
    )
    gate_evidence_by_stage.setdefault(selected_stage_name, {})["metric_drift_effective_cap"] = float(
        selected.get("metric_drift_effective_cap", surrogate_drift_effective_cap_for_eval)
    )
    gate_evidence_by_stage.setdefault(selected_stage_name, {})["selection_quality_score_raw_drift"] = float(
        selected.get("selection_quality_score_raw_drift", 0.0)
    )
    gate_evidence_by_stage.setdefault(selected_stage_name, {})["compression_refine_applied"] = bool(
        selected.get("compression_refine_applied", False)
    )
    gate_evidence_by_stage.setdefault(selected_stage_name, {})["compression_refine_trials"] = int(
        selected.get("compression_refine_trials", 0)
    )
    gate_evidence_by_stage.setdefault(selected_stage_name, {})["compression_refine_reaction_delta"] = int(
        selected.get("compression_refine_reaction_delta", 0)
    )
    gate_evidence_by_stage.setdefault(selected_stage_name, {})["compression_refine_species_delta"] = int(
        selected.get("compression_refine_species_delta", 0)
    )
    gate_evidence_by_stage.setdefault(selected_stage_name, {})["compression_refine_mode_effective"] = str(
        selected.get("compression_refine_mode_effective", "none")
    )
    gate_evidence_by_stage.setdefault(selected_stage_name, {})["compression_refine_guard_passed"] = bool(
        selected.get("compression_refine_guard_passed", True)
    )
    gate_evidence_by_stage.setdefault(selected_stage_name, {})["evaluation_contract_version"] = str(
        selected.get("evaluation_contract_version", contract_cfg.get("version", "v1"))
    )
    gate_evidence_by_stage.setdefault(selected_stage_name, {})["metric_taxonomy_profile_effective"] = str(
        selected.get("metric_taxonomy_profile_effective", metric_taxonomy_resolved.get("profile", "legacy_builtin"))
    )
    gate_evidence_by_stage.setdefault(selected_stage_name, {})["diagnostic_schema_ok"] = bool(
        selected.get("diagnostic_schema_ok", True)
    )
    selected_gate_evidence.update(selected_blockers)
    pareto_candidates = [dict(r) for r in list(selection_result.get("pareto_candidates") or [])]
    for row in pareto_candidates:
        row.pop("_floors", None)
        row.pop("_selection_max_mean_rel", None)
        row.pop("_selection_use_raw_drift", None)
        row.pop("_selection_raw_drift_cap", None)
    candidate_trend = [
        {
            "stage": str(row.get("stage")),
            "species_after": int(row.get("species_after", 0)),
            "gas_species_after": int(row.get("gas_species_after", 0)),
            "surface_species_after": int(row.get("surface_species_after", 0)),
            "reactions_after": int(row.get("reactions_after", 0)),
            "gas_reactions_after": (
                None
                if row.get("gas_reactions_after") is None
                else int(row.get("gas_reactions_after", 0))
            ),
            "surface_reactions_after": (
                None
                if row.get("surface_reactions_after") is None
                else int(row.get("surface_reactions_after", 0))
            ),
            "overall_candidates": int(row.get("overall_candidates", 0)),
            "overall_selected": int(row.get("overall_selected", 0)),
            "overall_select_ratio": float(row.get("overall_select_ratio", 0.0)),
            "metric_drift_raw": float(row.get("metric_drift_raw", row.get("metric_drift_effective", 1.0))),
            "metric_drift_effective_cap": float(
                row.get("metric_drift_effective_cap", surrogate_drift_effective_cap_for_eval)
            ),
            "mean_rel_diff": float(row.get("mean_rel_diff", 0.0)),
            "compression_refine_applied": bool(row.get("compression_refine_applied", False)),
            "compression_refine_trials": int(row.get("compression_refine_trials", 0)),
            "compression_refine_reaction_delta": int(row.get("compression_refine_reaction_delta", 0)),
            "compression_refine_species_delta": int(row.get("compression_refine_species_delta", 0)),
            "compression_refine_mode_effective": str(row.get("compression_refine_mode_effective", "none")),
            "compression_refine_guard_passed": bool(row.get("compression_refine_guard_passed", True)),
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
            "selection_quality_score_raw_drift": float(row.get("selection_quality_score_raw_drift", 0.0)),
            "selection_pool_kind": str(row.get("selection_pool_kind", "all")),
            "structure_deficit_score": float(row.get("structure_deficit_score", 0.0)),
            "pooling_candidate_count": int(row.get("pooling_candidate_count", 0)),
            "pooling_candidate_unique_count": int(row.get("pooling_candidate_unique_count", 0)),
            "pooling_candidate_selected_backend": str(row.get("pooling_candidate_selected_backend", "")),
            "pooling_candidate_selected_source": str(row.get("pooling_candidate_selected_source", "backend")),
            "pooling_candidate_selected_coverage_proxy": float(
                row.get("pooling_candidate_selected_coverage_proxy", 0.0)
            ),
            "pooling_candidate_selected_dynamics_recon_error": float(
                row.get("pooling_candidate_selected_dynamics_recon_error", 0.0)
            ),
            "pooling_candidate_selected_max_cluster_size_ratio": float(
                row.get("pooling_candidate_selected_max_cluster_size_ratio", 0.0)
            ),
            "pooling_candidate_scores": list(row.get("pooling_candidate_scores") or []),
            "balance_margin": float(row.get("balance_margin", 0.0)),
            "balance_dynamic_applied": bool(row.get("balance_dynamic_applied", False)),
            "balance_dynamic_complexity": float(row.get("balance_dynamic_complexity", 0.0)),
            "rs_upper_effective": float(row.get("rs_upper_effective", 0.0)),
            "active_cov_effective_floor": float(row.get("active_cov_effective_floor", 0.0)),
            "mandatory_total_metric_count": int(row.get("mandatory_total_metric_count", 0)),
            "valid_mandatory_metric_count": int(row.get("valid_mandatory_metric_count", 0)),
            "mandatory_validity_passed": bool(row.get("mandatory_validity_passed", True)),
            "inactive_mandatory_metric_count": int(row.get("inactive_mandatory_metric_count", 0)),
            "active_invalid_mandatory_metric_count": int(row.get("active_invalid_mandatory_metric_count", 0)),
            "mandatory_metric_case_pass_rates": dict(row.get("mandatory_metric_case_pass_rates") or {}),
            "mandatory_metric_valid_case_pass_min_effective": float(
                row.get("mandatory_metric_valid_case_pass_min_effective", 0.0)
            ),
            "mandatory_metric_validity_mode_effective": str(
                row.get("mandatory_metric_validity_mode_effective", "case_pass_rate")
            ),
            "mandatory_total_gate_unit_count": int(row.get("mandatory_total_gate_unit_count", 0)),
            "valid_mandatory_gate_unit_count": int(row.get("valid_mandatory_gate_unit_count", 0)),
            "valid_mandatory_gate_unit_count_case_rate": int(
                row.get("valid_mandatory_gate_unit_count_case_rate", row.get("valid_mandatory_gate_unit_count", 0))
            ),
            "valid_mandatory_gate_unit_count_coverage": int(
                row.get("valid_mandatory_gate_unit_count_coverage", row.get("valid_mandatory_gate_unit_count", 0))
            ),
            "mandatory_gate_unit_valid_count_shadow_evaluable_ratio": int(
                row.get("mandatory_gate_unit_valid_count_shadow_evaluable_ratio", 0)
            ),
            "mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective": float(
                row.get("mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective", 0.25)
            ),
            "mandatory_validity_basis_effective": str(
                row.get("mandatory_validity_basis_effective", "coverage_evaluable")
            ),
            "mandatory_quality_gate_unit_count": int(row.get("mandatory_quality_gate_unit_count", 0)),
            "mandatory_quality_metric_count": int(row.get("mandatory_quality_metric_count", 0)),
            "active_invalid_mandatory_gate_unit_keys": list(
                row.get("active_invalid_mandatory_gate_unit_keys") or []
            ),
            "mandatory_gate_unit_case_pass_rates": dict(row.get("mandatory_gate_unit_case_pass_rates") or {}),
            "mandatory_gate_unit_evaluable_case_rates": dict(
                row.get("mandatory_gate_unit_evaluable_case_rates") or {}
            ),
            "mandatory_gate_unit_mode_effective": str(
                row.get("mandatory_gate_unit_mode_effective", "species_family_quorum")
            ),
            "mandatory_species_family_score_mode_effective": str(
                row.get("mandatory_species_family_score_mode_effective", "uniform")
            ),
            "mandatory_quality_scope_effective": str(
                row.get("mandatory_quality_scope_effective", "valid_only")
            ),
            "mandatory_tail_scope_effective": str(
                row.get("mandatory_tail_scope_effective", "quality_scope")
            ),
            "mandatory_species_family_case_pass_min_effective": float(
                row.get("mandatory_species_family_case_pass_min_effective", 0.67)
            ),
            "pass_rate_mandatory_case": float(row.get("pass_rate_mandatory_case", row.get("pass_rate", 0.0))),
            "pass_rate_mandatory_case_all_units": float(
                row.get(
                    "pass_rate_mandatory_case_all_units",
                    row.get("pass_rate_mandatory_case", row.get("pass_rate", 0.0)),
                )
            ),
            "pass_rate_mandatory_case_all_required": float(
                row.get("pass_rate_mandatory_case_all_required", row.get("pass_rate_mandatory_case", row.get("pass_rate", 0.0)))
            ),
            "pass_rate_mandatory_case_ratio_mean": float(
                row.get("pass_rate_mandatory_case_ratio_mean", row.get("pass_rate_mandatory_case", row.get("pass_rate", 0.0)))
            ),
            "pass_rate_mandatory_case_all_required_all_units": float(
                row.get(
                    "pass_rate_mandatory_case_all_required_all_units",
                    row.get("pass_rate_mandatory_case_all_required", row.get("pass_rate_mandatory_case", row.get("pass_rate", 0.0))),
                )
            ),
            "pass_rate_mandatory_case_ratio_mean_all_units": float(
                row.get(
                    "pass_rate_mandatory_case_ratio_mean_all_units",
                    row.get("pass_rate_mandatory_case_ratio_mean", row.get("pass_rate_mandatory_case", row.get("pass_rate", 0.0))),
                )
            ),
            "pass_rate_all_metric_legacy": float(
                row.get("pass_rate_all_metric_legacy", row.get("pass_rate", 0.0))
            ),
            "mandatory_case_mode_effective": str(row.get("mandatory_case_mode_effective", "ratio_mean")),
            "mandatory_case_unit_weight_mode_effective": str(
                row.get("mandatory_case_unit_weight_mode_effective", "uniform")
            ),
            "pass_rate_optional_case": float(row.get("pass_rate_optional_case", row.get("pass_rate", 0.0))),
            "pass_rate_optional_metric_mean": float(
                row.get("pass_rate_optional_metric_mean", row.get("pass_rate", 0.0))
            ),
            "mean_rel_diff_mandatory": float(row.get("mean_rel_diff_mandatory", row.get("mean_rel_diff", 0.0))),
            "mean_rel_diff_mandatory_all_units": float(
                row.get("mean_rel_diff_mandatory_all_units", row.get("mean_rel_diff_mandatory", row.get("mean_rel_diff", 0.0)))
            ),
            "mean_rel_diff_mandatory_raw": float(
                row.get("mean_rel_diff_mandatory_raw", row.get("mean_rel_diff_mandatory", row.get("mean_rel_diff", 0.0)))
            ),
            "mean_rel_diff_mandatory_family_weighted": float(
                row.get(
                    "mean_rel_diff_mandatory_family_weighted",
                    row.get("mean_rel_diff_mandatory", row.get("mean_rel_diff", 0.0)),
                )
            ),
            "mean_rel_diff_mandatory_winsorized": float(
                row.get(
                    "mean_rel_diff_mandatory_winsorized",
                    row.get("mean_rel_diff_mandatory", row.get("mean_rel_diff", 0.0)),
                )
            ),
            "mandatory_rel_outlier_ratio": float(row.get("mandatory_rel_outlier_ratio", 0.0)),
            "mandatory_rel_outlier_ratio_all_units": float(
                row.get("mandatory_rel_outlier_ratio_all_units", row.get("mandatory_rel_outlier_ratio", 0.0))
            ),
            "mandatory_rel_outlier_ratio_max_effective": float(
                row.get("mandatory_rel_outlier_ratio_max_effective", 0.20)
            ),
            "mandatory_rel_diff_p95": float(row.get("mandatory_rel_diff_p95", 0.0)),
            "mandatory_rel_diff_p95_all_units": float(
                row.get("mandatory_rel_diff_p95_all_units", row.get("mandatory_rel_diff_p95", 0.0))
            ),
            "mandatory_tail_guard_passed": bool(row.get("mandatory_tail_guard_passed", True)),
            "mandatory_tail_guard_triggered": bool(row.get("mandatory_tail_guard_triggered", False)),
            "mandatory_tail_guard_hard_applied": bool(row.get("mandatory_tail_guard_hard_applied", False)),
            "mandatory_tail_guard_mode_effective": str(
                row.get("mandatory_tail_guard_mode_effective", "p95")
            ),
            "mandatory_tail_guard_policy_effective": str(
                row.get("mandatory_tail_guard_policy_effective", "conditional_hard")
            ),
            "mandatory_tail_activation_ratio_min_effective": float(
                row.get("mandatory_tail_activation_ratio_min_effective", 0.10)
            ),
            "mandatory_tail_exceed_ref_effective": str(
                row.get("mandatory_tail_exceed_ref_effective", "tail_max")
            ),
            "mandatory_tail_exceed_ratio": float(row.get("mandatory_tail_exceed_ratio", 0.0)),
            "mandatory_tail_rel_diff_max_effective": float(
                row.get("mandatory_tail_rel_diff_max_effective", 1.50)
            ),
            "mandatory_quality_scope_empty": bool(row.get("mandatory_quality_scope_empty", False)),
            "mandatory_mean_aggregation_effective": str(row.get("mandatory_mean_aggregation_effective", "raw")),
            "mandatory_mean_mode_effective": str(row.get("mandatory_mean_mode_effective", "winsorized")),
            "mean_rel_diff_optional": float(row.get("mean_rel_diff_optional", row.get("mean_rel_diff", 0.0))),
            "mean_rel_diff_all_metric_legacy": float(
                row.get("mean_rel_diff_all_metric_legacy", row.get("mean_rel_diff", 0.0))
            ),
            "error_gate_score": float(row.get("error_gate_score", 0.0)),
            "error_gate_passed": bool(row.get("error_gate_passed", row.get("gate_passed", False))),
            "coverage_gate_passed": bool(row.get("coverage_gate_passed", row.get("mandatory_validity_passed", True))),
            "mandatory_quality_passed": bool(row.get("mandatory_quality_passed", row.get("mandatory_error_passed", True))),
            "optional_quality_passed": bool(row.get("optional_quality_passed", row.get("optional_error_passed", True))),
            "mandatory_error_passed": bool(row.get("mandatory_error_passed", row.get("error_gate_passed", True))),
            "optional_error_passed": bool(row.get("optional_error_passed", True)),
            "mandatory_error_include_validity_effective": bool(
                row.get("mandatory_error_include_validity_effective", False)
            ),
            "error_fail_reason_primary": str(row.get("error_fail_reason_primary", "none")),
            "evaluation_contract_version": str(
                row.get("evaluation_contract_version", contract_cfg.get("version", "v1"))
            ),
            "metric_taxonomy_profile_effective": str(
                row.get(
                    "metric_taxonomy_profile_effective",
                    metric_taxonomy_resolved.get("profile", "legacy_builtin"),
                )
            ),
            "diagnostic_schema_ok": bool(row.get("diagnostic_schema_ok", True)),
            "effective_metric_count": int(row.get("effective_metric_count", row.get("qoi_metrics_count", 0))),
            "suppressed_low_signal_metric_count": int(row.get("suppressed_low_signal_metric_count", 0)),
            "metric_clip_guardrail_trigger_ratio": float(row.get("metric_clip_guardrail_trigger_ratio", 0.0)),
            "replay_health_trust_invalid": bool(row.get("replay_health_trust_invalid", False)),
            "structure_feedback_multiplier": float(row.get("structure_feedback_multiplier", 1.0)),
            "primary_blocker_layer": str(row.get("primary_blocker_layer", "none")),
            "secondary_blockers": list(row.get("secondary_blockers") or []),
            "validity_fail_reason_primary": str(row.get("validity_fail_reason_primary", "none")),
            "timing_stage_s": float(row.get("timing_stage_s", 0.0)),
            "timing_pooling_fit_s": float(row.get("timing_pooling_fit_s", 0.0)),
            "timing_bridge_s": float(row.get("timing_bridge_s", 0.0)),
            "timing_surrogate_eval_s": float(row.get("timing_surrogate_eval_s", 0.0)),
            "timing_physical_gate_s": float(row.get("timing_physical_gate_s", 0.0)),
            "timing_projection_s": float(row.get("timing_projection_s", 0.0)),
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
    timing_total_s = float(pytime.perf_counter() - total_started)
    run_finished_at = _utc_now_iso()
    runtime_meta["finished_at"] = run_finished_at
    selected["timing_total_s"] = timing_total_s
    selected["evaluation_contract_version"] = str(
        selected.get("evaluation_contract_version", contract_cfg.get("version", "v1"))
    )
    selected["metric_taxonomy_profile_effective"] = str(
        selected.get("metric_taxonomy_profile_effective", metric_taxonomy_resolved.get("profile", "legacy_builtin"))
    )
    selected["diagnostic_schema_ok"] = bool(selected.get("diagnostic_schema_ok", True))

    summary_payload = {
        "gate_passed": bool(selected["gate_passed"]),
        "hard_ban_violations": int(selected["hard_ban_violations"]),
        "primary_blocker_layer": str(selected.get("primary_blocker_layer", "none")),
        "secondary_blockers": list(selected.get("secondary_blockers") or []),
        "validity_fail_reason_primary": str(selected.get("validity_fail_reason_primary", "none")),
        "error_fail_reason_primary": str(selected.get("error_fail_reason_primary", "none")),
        "evaluation_contract_version": str(
            selected.get("evaluation_contract_version", contract_cfg.get("version", "v1"))
        ),
        "metric_taxonomy_profile_effective": str(
            selected.get("metric_taxonomy_profile_effective", metric_taxonomy_resolved.get("profile", "legacy_builtin"))
        ),
        "diagnostic_schema_ok": bool(selected.get("diagnostic_schema_ok", True)),
        "reduction_mode": reduction_mode,
        "timing_total_s": timing_total_s,
        "timing_stage_s": timing_stage_s,
        "timing_pooling_fit_s": timing_pooling_fit_s,
        "timing_bridge_s": timing_bridge_s,
        "timing_surrogate_eval_s": timing_surrogate_eval_s,
        "timing_physical_gate_s": timing_physical_gate_s,
        "timing_projection_s": timing_projection_s,
        "runtime": runtime_meta,
        "config_hash": runtime_meta.get("config_hash"),
        "git_commit": runtime_meta.get("git_commit"),
        "pid": runtime_meta.get("pid"),
        "started_at": runtime_meta.get("started_at"),
        "finished_at": runtime_meta.get("finished_at"),
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
        "evaluation_contract": dict(contract_cfg),
        "metric_taxonomy": {
            "source": str((eval_cfg.get("metric_taxonomy") or {}).get("source", "legacy_builtin")),
            "profile": str(metric_taxonomy_resolved.get("profile", "legacy_builtin")),
            "path": (eval_cfg.get("metric_taxonomy") or {}).get("path"),
        },
        "non_regression": dict(non_regression_cfg),
        "phase_context": phase_context,
        "domain_counts": {
            "species_before": species_domain_counts_before,
            "reactions_before": reaction_domain_counts_before,
            "selected_species_after": {
                "gas": int(selected.get("gas_species_after", 0)),
                "surface": int(selected.get("surface_species_after", 0)),
            },
            "selected_reactions_after": {
                "gas": (
                    None
                    if selected.get("gas_reactions_after") is None
                    else int(selected.get("gas_reactions_after", 0))
                ),
                "surface": (
                    None
                    if selected.get("surface_reactions_after") is None
                    else int(selected.get("surface_reactions_after", 0))
                ),
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
            "candidate_count": int(selected.get("pooling_candidate_count", selected_pool_metrics.get("candidate_count", 0))),
            "candidate_unique_count": int(
                selected.get("pooling_candidate_unique_count", selected_pool_metrics.get("candidate_unique_count", 0))
            ),
            "candidate_selected_backend": str(
                selected.get("pooling_candidate_selected_backend", selected_pool_metrics.get("candidate_selected_backend", ""))
            ),
            "candidate_selected_source": str(
                selected.get("pooling_candidate_selected_source", selected_pool_metrics.get("candidate_selected_source", "backend"))
            ),
            "candidate_selected_coverage_proxy": float(
                selected.get(
                    "pooling_candidate_selected_coverage_proxy",
                    selected_pool_metrics.get("candidate_selected_coverage_proxy", 0.0),
                )
            ),
            "candidate_selected_dynamics_recon_error": float(
                selected.get(
                    "pooling_candidate_selected_dynamics_recon_error",
                    selected_pool_metrics.get("candidate_selected_dynamics_recon_error", 0.0),
                )
            ),
            "candidate_selected_max_cluster_size_ratio": float(
                selected.get(
                    "pooling_candidate_selected_max_cluster_size_ratio",
                    selected_pool_metrics.get("candidate_selected_max_cluster_size_ratio", 0.0),
                )
            ),
            "candidate_scores": list(
                selected.get("pooling_candidate_scores", selected_pool_metrics.get("candidate_scores", [])) or []
            ),
            "selected_stage": selected_stage_name,
            "mapping_fallback_enabled": bool(pooling_mapping_fallback_enabled),
            "mapping_fallback_triggered": bool(pooling_mapping_fallback_reasons),
            "mapping_fallback_reasons": pooling_mapping_fallback_reasons,
            "bridge_enabled": bool(pooling_bridge_enabled),
            "bridge_mode": str(pooling_bridge_mode_cfg),
        }
        summary_payload["pooling_artifact_path"] = str(
            pooling_stage_artifacts.get(selected_stage_name, selected.get("pooling_artifact_path") or "")
        )

    diagnostic_schema_ok = validate_summary_schema(
        summary_payload,
        strict=bool(contract_cfg.get("diagnostic_schema_strict", False)),
    )
    if not diagnostic_schema_ok and not bool(contract_cfg.get("diagnostic_schema_strict", False)):
        print("[WARN] diagnostic schema missing keys", file=sys.stderr)
    summary_payload["diagnostic_schema_ok"] = bool(diagnostic_schema_ok)
    summary_payload["selected_metrics"]["diagnostic_schema_ok"] = bool(diagnostic_schema_ok)
    summary_payload["gate_evidence"]["selected_stage_evidence"]["diagnostic_schema_ok"] = bool(
        diagnostic_schema_ok
    )

    runtime_guard.check("before_write_report")
    write_report(
        report_dir,
        run_id=args.run_id,
        stage_rows=stage_rows,
        selected_stage=str(selected["stage"]),
        summary_payload=summary_payload,
    )
    runtime_guard.mark_progress("report_written")

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    runtime_guard.release()


if __name__ == "__main__":
    main()
