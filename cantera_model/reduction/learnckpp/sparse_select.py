from __future__ import annotations

from typing import Any

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _estimate_rates(nu_cand: np.ndarray, ydot_target: np.ndarray) -> np.ndarray:
    if nu_cand.shape[1] == 0:
        return np.zeros((ydot_target.shape[0], 0), dtype=float)
    # ydot ~= rates @ nu_cand.T
    proj = np.linalg.pinv(nu_cand.T)
    return np.asarray(ydot_target, dtype=float) @ np.asarray(proj, dtype=float)


def _normalize_positive(x: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(x, dtype=float), 0.0)
    max_val = float(np.max(arr)) if arr.size else 0.0
    if max_val <= 0.0:
        return np.zeros_like(arr)
    return arr / max_val


def _keep_from_ratio(scores: np.ndarray, ratio: float) -> np.ndarray:
    n = scores.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool)
    k = int(round(float(np.clip(ratio, 0.0, 1.0)) * n))
    k = max(1, min(n, k))
    order = np.argsort(-scores)
    keep = np.zeros((n,), dtype=bool)
    keep[order[:k]] = True
    return keep


def _keep_from_quantile(scores: np.ndarray, quantile: float) -> np.ndarray:
    n = scores.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=bool)
    q = float(np.clip(quantile, 0.0, 1.0))
    th = float(np.quantile(scores, q))
    keep = scores >= th
    if not np.any(keep):
        keep[int(np.argmax(scores))] = True
    return keep


def _enforce_min_keep_count(keep: np.ndarray, scores: np.ndarray, min_keep_count: int) -> np.ndarray:
    out = np.asarray(keep, dtype=bool).copy()
    n = out.shape[0]
    if n == 0:
        return out
    mk = int(max(1, min(int(min_keep_count), n)))
    current = int(np.sum(out))
    if current >= mk:
        return out

    order = np.argsort(-np.asarray(scores, dtype=float))
    for idx in order:
        if not bool(out[int(idx)]):
            out[int(idx)] = True
            current += 1
        if current >= mk:
            break
    return out


def _coverage_metrics(
    keep: np.ndarray,
    nu_cand: np.ndarray,
    cluster_weights: np.ndarray,
    essential_cluster_mask: np.ndarray,
) -> tuple[float, float]:
    keep_arr = np.asarray(keep, dtype=bool)
    nu_arr = np.asarray(nu_cand, dtype=float)
    n_clusters = int(nu_arr.shape[0])
    if n_clusters <= 0:
        return 0.0, 1.0
    if np.sum(keep_arr) <= 0:
        active_clusters = np.zeros((n_clusters,), dtype=bool)
    else:
        active_clusters = np.any(np.abs(nu_arr[:, keep_arr]) > 1.0e-12, axis=1)

    weights = np.asarray(cluster_weights, dtype=float)
    if weights.shape != (n_clusters,):
        fixed = np.zeros((n_clusters,), dtype=float)
        if weights.ndim == 1:
            copy_n = min(n_clusters, int(weights.shape[0]))
            if copy_n > 0:
                fixed[:copy_n] = np.maximum(weights[:copy_n], 0.0)
        weights = fixed
    w_sum = float(np.sum(weights))
    if w_sum > 0.0:
        weighted_cov = float(np.sum(weights[active_clusters])) / w_sum
    else:
        weighted_cov = float(np.sum(active_clusters)) / float(max(1, n_clusters))

    essential = np.asarray(essential_cluster_mask, dtype=bool)
    if essential.shape != (n_clusters,):
        fixed_e = np.zeros((n_clusters,), dtype=bool)
        if essential.ndim == 1:
            copy_n = min(n_clusters, int(essential.shape[0]))
            if copy_n > 0:
                fixed_e[:copy_n] = essential[:copy_n]
        essential = fixed_e
    n_essential = int(np.sum(essential))
    if n_essential <= 0:
        essential_cov = 1.0
    else:
        essential_active = np.logical_and(active_clusters, essential)
        essential_weights = weights[essential]
        if essential_weights.size > 0 and float(np.sum(essential_weights)) > 0.0:
            essential_cov = float(np.sum(weights[essential_active])) / float(np.sum(essential_weights))
        else:
            essential_cov = float(np.sum(essential_active)) / float(max(1, n_essential))
    return float(np.clip(weighted_cov, 0.0, 1.0)), float(np.clip(essential_cov, 0.0, 1.0))


def _active_cluster_mask(keep: np.ndarray, nu_cand: np.ndarray) -> np.ndarray:
    keep_arr = np.asarray(keep, dtype=bool)
    nu_arr = np.asarray(nu_cand, dtype=float)
    n_clusters = int(nu_arr.shape[0]) if nu_arr.ndim == 2 else 0
    if n_clusters <= 0 or int(np.sum(keep_arr)) <= 0:
        return np.zeros((max(0, n_clusters),), dtype=bool)
    return np.any(np.abs(nu_arr[:, keep_arr]) > 1.0e-12, axis=1)


def enforce_active_cluster_coverage(
    *,
    keep: np.ndarray,
    importance: np.ndarray,
    nu_cand: np.ndarray,
    min_active_clusters: int,
    max_steps: int,
    max_keep_count: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    keep_arr = np.asarray(keep, dtype=bool).copy()
    imp = np.asarray(importance, dtype=float)
    nu_arr = np.asarray(nu_cand, dtype=float)
    n_cand = int(keep_arr.shape[0])
    if nu_arr.ndim != 2 or n_cand == 0:
        return keep_arr, {
            "enabled": False,
            "applied": False,
            "steps": 0,
            "active_clusters": 0,
            "target_active_clusters": 0,
        }

    n_clusters = int(nu_arr.shape[0])
    target = int(max(0, min(int(min_active_clusters), n_clusters)))
    if target <= 0:
        active = _active_cluster_mask(keep_arr, nu_arr)
        return keep_arr, {
            "enabled": False,
            "applied": False,
            "steps": 0,
            "active_clusters": int(np.sum(active)),
            "target_active_clusters": target,
        }

    cap = int(n_cand if max_keep_count is None else max(1, min(int(max_keep_count), n_cand)))
    steps_max = int(max(0, max_steps))
    steps = 0
    applied = False

    while steps < steps_max:
        active = _active_cluster_mask(keep_arr, nu_arr)
        active_count = int(np.sum(active))
        if active_count >= target:
            break
        uncovered = np.where(~active)[0]
        if uncovered.size == 0:
            break

        best_add = -1
        best_gain = -1
        best_score = -float("inf")
        for idx in np.where(~keep_arr)[0]:
            touched = np.abs(nu_arr[uncovered, int(idx)]) > 1.0e-12
            gain = int(np.sum(touched))
            score = float(imp[int(idx)])
            if gain > best_gain or (gain == best_gain and score > best_score):
                best_gain = gain
                best_score = score
                best_add = int(idx)
        if best_add < 0 or best_gain <= 0:
            break

        if int(np.sum(keep_arr)) < cap:
            keep_arr[best_add] = True
            applied = True
            steps += 1
            continue

        kept_idxs = [int(i) for i in np.where(keep_arr)[0]]
        best_remove = -1
        best_trial_count = active_count
        best_trial_score = -float("inf")
        for rem in sorted(kept_idxs, key=lambda i: float(imp[i])):
            trial = keep_arr.copy()
            trial[rem] = False
            trial[best_add] = True
            trial_active = _active_cluster_mask(trial, nu_arr)
            trial_count = int(np.sum(trial_active))
            trial_score = float(imp[best_add] - imp[rem])
            if trial_count > best_trial_count or (trial_count == best_trial_count and trial_score > best_trial_score):
                best_trial_count = trial_count
                best_trial_score = trial_score
                best_remove = rem
        if best_remove < 0 or best_trial_count <= active_count:
            break

        keep_arr[best_remove] = False
        keep_arr[best_add] = True
        applied = True
        steps += 1

    active = _active_cluster_mask(keep_arr, nu_arr)
    return keep_arr, {
        "enabled": True,
        "applied": applied,
        "steps": int(steps),
        "active_clusters": int(np.sum(active)),
        "target_active_clusters": int(target),
        "max_keep_count": int(cap),
    }


def coverage_aware_postselect(
    *,
    keep: np.ndarray,
    importance: np.ndarray,
    nu_cand: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    post_cfg = dict(cfg or {})
    enabled = bool(post_cfg.get("enabled", False))
    mode = str(post_cfg.get("mode", "binary")).strip().lower() or "binary"
    keep_arr = np.asarray(keep, dtype=bool).copy()
    imp = np.asarray(importance, dtype=float)
    nu_arr = np.asarray(nu_cand, dtype=float)
    n_cand = int(keep_arr.shape[0])
    if not enabled or n_cand == 0:
        weighted_cov, essential_cov = _coverage_metrics(
            keep_arr,
            nu_arr,
            np.asarray(post_cfg.get("cluster_weights") or np.zeros((nu_arr.shape[0],)), dtype=float),
            np.asarray(post_cfg.get("essential_cluster_mask") or np.zeros((nu_arr.shape[0],), dtype=bool), dtype=bool),
        )
        return keep_arr, {
            "enabled": enabled,
            "mode": mode,
            "applied": False,
            "weighted_coverage": weighted_cov,
            "essential_coverage": essential_cov,
            "target_weighted_coverage": float(post_cfg.get("target_weighted_coverage", 0.0)),
            "target_essential_coverage": float(post_cfg.get("target_essential_coverage", 0.0)),
        }

    cluster_weights = np.asarray(post_cfg.get("cluster_weights") or np.zeros((nu_arr.shape[0],)), dtype=float)
    essential_mask = np.asarray(post_cfg.get("essential_cluster_mask") or np.zeros((nu_arr.shape[0],), dtype=bool), dtype=bool)
    target_weighted = float(post_cfg.get("target_weighted_coverage", 0.0))
    target_essential = float(post_cfg.get("target_essential_coverage", 0.0))
    max_keep_count = int(max(1, min(int(post_cfg.get("max_keep_count", n_cand)), n_cand)))

    def _covered(weighted_cov: float, essential_cov: float) -> bool:
        checks: list[bool] = []
        if mode in {"weighted", "hybrid"}:
            checks.append(weighted_cov >= target_weighted)
        if mode in {"essential", "hybrid"} and target_essential > 0.0:
            checks.append(essential_cov >= target_essential)
        if mode == "binary":
            checks.append(True)
        return all(checks) if checks else True

    weighted_cov, essential_cov = _coverage_metrics(keep_arr, nu_arr, cluster_weights, essential_mask)
    if _covered(weighted_cov, essential_cov):
        return keep_arr, {
            "enabled": True,
            "mode": mode,
            "applied": False,
            "weighted_coverage": weighted_cov,
            "essential_coverage": essential_cov,
            "target_weighted_coverage": target_weighted,
            "target_essential_coverage": target_essential,
        }

    imp_norm = np.maximum(imp, 0.0)
    imp_max = float(np.max(imp_norm)) if imp_norm.size else 0.0
    if imp_max > 0.0:
        imp_norm = imp_norm / imp_max

    applied = False
    while not _covered(weighted_cov, essential_cov):
        candidate_idxs = [int(i) for i in np.where(~keep_arr)[0]]
        if not candidate_idxs:
            break

        best_idx = -1
        best_gain = -1.0e18
        best_keep = keep_arr
        for idx in candidate_idxs:
            trial = keep_arr.copy()
            if int(np.sum(trial)) < max_keep_count:
                trial[idx] = True
            else:
                kept = [int(i) for i in np.where(trial)[0]]
                if not kept:
                    continue
                replaced = False
                for remove_idx in sorted(kept, key=lambda x: float(imp_norm[x])):
                    trial2 = trial.copy()
                    trial2[remove_idx] = False
                    trial2[idx] = True
                    replaced = True
                    trial = trial2
                    break
                if not replaced:
                    continue
            trial_w, trial_e = _coverage_metrics(trial, nu_arr, cluster_weights, essential_mask)
            gain = (trial_w - weighted_cov) + (0.5 * (trial_e - essential_cov)) + (0.05 * float(imp_norm[idx]))
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                best_keep = trial
        if best_idx < 0 or best_gain <= 1.0e-12:
            break
        keep_arr = best_keep
        weighted_cov, essential_cov = _coverage_metrics(keep_arr, nu_arr, cluster_weights, essential_mask)
        applied = True

    return keep_arr, {
        "enabled": True,
        "mode": mode,
        "applied": applied,
        "weighted_coverage": weighted_cov,
        "essential_coverage": essential_cov,
        "target_weighted_coverage": target_weighted,
        "target_essential_coverage": target_essential,
    }


def select_sparse_overall(
    nu_cand: np.ndarray,
    ydot_target: np.ndarray,
    features: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    nu_arr = np.asarray(nu_cand, dtype=float)
    ydot_arr = np.asarray(ydot_target, dtype=float)
    feat_arr = np.asarray(features, dtype=float)

    if nu_arr.ndim != 2:
        raise ValueError("nu_cand must be 2-D")
    if ydot_arr.ndim != 2:
        raise ValueError("ydot_target must be 2-D")
    if feat_arr.ndim != 2:
        raise ValueError("features must be 2-D")
    if ydot_arr.shape[1] != nu_arr.shape[0]:
        raise ValueError("ydot_target cluster dimension must match nu_cand rows")
    if ydot_arr.shape[0] != feat_arr.shape[0]:
        raise ValueError("features rows must match ydot_target rows")

    n_cand = nu_arr.shape[1]
    if n_cand == 0:
        return np.zeros((0,), dtype=bool), {
            "status": "empty_candidates",
            "method": str((cfg or {}).get("method", "hard_concrete")),
            "method_used": "none",
            "fallback": False,
            "keep_count": 0,
            "keep_ratio": 0.0,
            "importance": [],
            "rates_target": np.zeros((ydot_arr.shape[0], 0), dtype=float),
        }

    method = str((cfg or {}).get("method", "hard_concrete"))
    lambda_l0 = float((cfg or {}).get("lambda_l0", 1.0e-3))
    target_keep_ratio = float((cfg or {}).get("target_keep_ratio", 0.3))
    min_keep_count_cfg = int((cfg or {}).get("min_keep_count", 1))
    min_active_clusters_cfg = int((cfg or {}).get("min_active_clusters", 0))
    coverage_aware_swap_steps = int((cfg or {}).get("coverage_aware_swap_steps", 8))
    force_fallback = bool((cfg or {}).get("force_fallback", False))
    seed = int((cfg or {}).get("seed", 0))
    rng = np.random.default_rng(seed)

    rates_target = _estimate_rates(nu_arr, ydot_arr)
    importance = np.mean(np.abs(rates_target), axis=0)
    importance = np.asarray(importance, dtype=float)

    status = "ok"
    method_used = method
    fallback = False
    probs = np.zeros((n_cand,), dtype=float)

    if force_fallback:
        status = "fallback_forced"
        fallback = True

    if not fallback:
        if method == "hard_concrete":
            logits = np.log(importance + 1.0e-12)
            logits = logits - float(np.mean(logits))
            scale = float(np.std(logits)) + 1.0e-6
            logits = logits / scale
            noise = rng.normal(0.0, 0.05, size=logits.shape)
            probs = _sigmoid(logits + noise - (lambda_l0 * 10.0))
            if not np.all(np.isfinite(probs)):
                status = "fallback_non_finite"
                fallback = True
        elif method == "l1_threshold":
            probs = _normalize_positive(importance)
        else:
            raise ValueError("select.method must be 'hard_concrete' or 'l1_threshold'")

    if fallback:
        fb = dict((cfg or {}).get("fallback") or {})
        fallback_method = str(fb.get("select_method", "l1_threshold"))
        method_used = fallback_method
        probs = _normalize_positive(importance)
        if fallback_method != "l1_threshold":
            status = "fallback_unknown_method"
        threshold_quantile = float(fb.get("threshold_quantile", 0.75))
        keep = _keep_from_quantile(probs, threshold_quantile)
    else:
        keep = _keep_from_ratio(probs, target_keep_ratio)

    if not np.any(keep):
        keep[int(np.argmax(importance))] = True
        status = "forced_one_candidate"
    keep = _enforce_min_keep_count(keep, importance, min_keep_count_cfg)
    coverage_post_cfg = dict((cfg or {}).get("coverage_postselect") or {})
    keep, coverage_meta = coverage_aware_postselect(
        keep=keep,
        importance=importance,
        nu_cand=nu_arr,
        cfg=coverage_post_cfg,
    )
    coverage_max_keep = int(coverage_post_cfg.get("max_keep_count", n_cand))
    keep, active_cluster_meta = enforce_active_cluster_coverage(
        keep=keep,
        importance=importance,
        nu_cand=nu_arr,
        min_active_clusters=min_active_clusters_cfg,
        max_steps=coverage_aware_swap_steps,
        max_keep_count=coverage_max_keep,
    )
    active_clusters = int(active_cluster_meta.get("active_clusters", 0))
    n_clusters = int(nu_arr.shape[0])
    active_cov = float(active_clusters) / float(max(1, n_clusters))

    score: dict[str, Any] = {
        "status": status,
        "method": method,
        "method_used": method_used,
        "fallback": bool(fallback),
        "keep_count": int(np.sum(keep)),
        "keep_ratio": float(np.mean(keep)),
        "target_keep_ratio": target_keep_ratio,
        "min_keep_count": int(max(1, min(min_keep_count_cfg, n_cand))),
        "coverage_postselect": coverage_meta,
        "active_cluster_coverage": active_cluster_meta,
        "active_species_coverage": float(active_cov),
        "min_active_clusters": int(max(0, min(min_active_clusters_cfg, n_clusters))),
        "weighted_active_species_coverage": float(coverage_meta.get("weighted_coverage", 0.0)),
        "essential_species_coverage": float(coverage_meta.get("essential_coverage", 1.0)),
        "importance": importance.tolist(),
        "rates_target": rates_target,
    }
    return np.asarray(keep, dtype=bool), score
