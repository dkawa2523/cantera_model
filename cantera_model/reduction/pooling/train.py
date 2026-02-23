from __future__ import annotations

from typing import Any

import numpy as np

from cantera_model.reduction.pooling.constraints import pooling_constraint_loss
from cantera_model.reduction.pooling.models import build_pooling_model, infer_assignment


def _hard_assign(s_prob: np.ndarray) -> np.ndarray:
    prob = np.asarray(s_prob, dtype=float)
    if prob.ndim != 2:
        raise ValueError("S_prob must be 2-D")
    n, k = prob.shape
    out = np.zeros((n, k), dtype=float)
    if n == 0 or k == 0:
        return out
    idx = np.argmax(prob, axis=1)
    out[np.arange(n), idx] = 1.0
    return out


def _repair_hard_violations(S: np.ndarray, hard_mask: np.ndarray, pair_cost: np.ndarray) -> np.ndarray:
    init = np.asarray(S, dtype=float)
    n = init.shape[0]
    preferred = np.argmax(init, axis=1) if init.size else np.zeros((n,), dtype=int)

    clusters: list[list[int]] = []
    assign = np.full((n,), -1, dtype=int)

    for i in range(n):
        best_c = -1
        best_score = float("inf")
        for c_idx, members in enumerate(clusters):
            if any(not bool(hard_mask[i, m]) for m in members):
                continue
            score = float(sum(float(pair_cost[i, m]) for m in members))
            # Prefer original cluster if tie.
            if c_idx == int(preferred[i]):
                score -= 1.0e-6
            if score < best_score:
                best_score = score
                best_c = c_idx

        if best_c < 0:
            clusters.append([i])
            assign[i] = len(clusters) - 1
        else:
            clusters[best_c].append(i)
            assign[i] = best_c

    out = np.zeros((n, len(clusters)), dtype=float)
    if n > 0:
        out[np.arange(n), assign] = 1.0
    return out


def _cluster_meta(S: np.ndarray, species_meta: list[dict[str, Any]]) -> list[dict[str, Any]]:
    s = np.asarray(S, dtype=float)
    out: list[dict[str, Any]] = []
    for c in range(s.shape[1]):
        idx = np.where(s[:, c] > 0.5)[0]
        names = [str(species_meta[i].get("name", f"sp{i}")) for i in idx]
        elems = sorted({e for i in idx for e in (species_meta[i].get("composition") or {}).keys()})
        out.append({"cluster_id": int(c), "members": names, "elements": elems})
    return out


def _enforce_min_clusters(S: np.ndarray, min_clusters: int) -> np.ndarray:
    s = np.asarray(S, dtype=float)
    if s.ndim != 2:
        raise ValueError("S must be 2-D")
    n, k = s.shape
    target = int(max(1, min(int(min_clusters), n)))
    if k >= target:
        return s

    out = s.copy()
    while out.shape[1] < target:
        sizes = np.sum(out > 0.5, axis=0)
        split_idx = int(np.argmax(sizes))
        members = np.where(out[:, split_idx] > 0.5)[0]
        if members.size <= 1:
            break
        mover = int(members[-1])
        out[mover, split_idx] = 0.0
        new_col = np.zeros((n, 1), dtype=float)
        new_col[mover, 0] = 1.0
        out = np.concatenate([out, new_col], axis=1)
    return out


def _cluster_coverage_proxy(S: np.ndarray, species_weights: np.ndarray) -> float:
    s = np.asarray(S, dtype=float)
    w = np.maximum(np.asarray(species_weights, dtype=float), 0.0)
    if s.ndim != 2 or w.shape != (s.shape[0],) or s.shape[1] <= 0:
        return 0.0
    cluster_mass = np.asarray(s.T @ w, dtype=float)
    total = float(np.sum(cluster_mass))
    if total <= 0.0:
        sizes = np.sum(s > 0.5, axis=0).astype(float)
        total_sizes = float(np.sum(sizes))
        if total_sizes <= 0.0:
            return 0.0
        cluster_mass = sizes
        total = total_sizes
    p = np.asarray(cluster_mass / total, dtype=float)
    return float(np.clip(1.0 - float(np.max(p)), 0.0, 1.0))


def _repair_coverage_target(
    S: np.ndarray,
    species_weights: np.ndarray,
    coverage_target: float,
    coverage_max_clusters: int,
) -> tuple[np.ndarray, float]:
    s = np.asarray(S, dtype=float)
    n, k = s.shape
    target = float(np.clip(coverage_target, 0.0, 1.0))
    if n <= 1 or k <= 0:
        return s, _cluster_coverage_proxy(s, species_weights)
    max_clusters = int(max(k, min(int(coverage_max_clusters), n)))
    out = s.copy()
    proxy = _cluster_coverage_proxy(out, species_weights)
    if proxy >= target:
        return out, proxy

    weights = np.maximum(np.asarray(species_weights, dtype=float), 0.0)
    while out.shape[1] < max_clusters and proxy < target:
        cluster_mass = np.asarray(out.T @ weights, dtype=float)
        split_idx = int(np.argmax(cluster_mass))
        members = np.where(out[:, split_idx] > 0.5)[0]
        if members.size <= 1:
            break
        mover = int(members[np.argmax(weights[members])])
        out[mover, split_idx] = 0.0
        new_col = np.zeros((n, 1), dtype=float)
        new_col[mover, 0] = 1.0
        out = np.concatenate([out, new_col], axis=1)
        proxy = _cluster_coverage_proxy(out, weights)
    return out, proxy


def _max_cluster_size_ratio(S: np.ndarray) -> float:
    s = np.asarray(S, dtype=float)
    if s.ndim != 2 or s.shape[0] <= 0 or s.shape[1] <= 0:
        return 0.0
    sizes = np.sum(s > 0.5, axis=0).astype(float)
    return float(np.max(sizes) / float(max(1, s.shape[0])))


def _repair_max_cluster_size(
    S: np.ndarray,
    species_weights: np.ndarray,
    max_cluster_size_ratio: float,
) -> tuple[np.ndarray, float]:
    s = np.asarray(S, dtype=float)
    if s.ndim != 2:
        raise ValueError("S must be 2-D")
    n, k = s.shape
    if n <= 1 or k <= 0:
        return s, _max_cluster_size_ratio(s)

    ratio_limit = float(np.clip(max_cluster_size_ratio, 0.0, 1.0))
    if ratio_limit <= 0.0:
        return s, _max_cluster_size_ratio(s)
    max_size = int(max(1, np.floor(ratio_limit * float(n))))

    out = s.copy()
    weights = np.maximum(np.asarray(species_weights, dtype=float), 0.0)
    if weights.shape != (n,):
        w = np.zeros((n,), dtype=float)
        copy_n = min(n, int(weights.shape[0])) if weights.ndim == 1 else 0
        if copy_n > 0 and weights.ndim == 1:
            w[:copy_n] = weights[:copy_n]
        weights = w

    while True:
        sizes = np.sum(out > 0.5, axis=0).astype(int)
        if sizes.size == 0:
            break
        split_idx = int(np.argmax(sizes))
        if int(sizes[split_idx]) <= max_size or out.shape[1] >= n:
            break
        members = np.where(out[:, split_idx] > 0.5)[0]
        if members.size <= 1:
            break
        mover = int(members[np.argmax(weights[members])]) if np.any(weights[members] > 0.0) else int(members[-1])
        out[mover, split_idx] = 0.0
        new_col = np.zeros((n, 1), dtype=float)
        new_col[mover, 0] = 1.0
        out = np.concatenate([out, new_col], axis=1)

    return out, _max_cluster_size_ratio(out)


def train_pooling_assignment(
    graph: dict[str, Any],
    features: np.ndarray,
    constraints: dict[str, Any],
    cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    c = dict(cfg or {})
    f = np.asarray(features, dtype=float)
    if f.ndim != 2:
        raise ValueError("features must be 2-D")

    hard_mask = np.asarray(constraints.get("hard_mask"), dtype=bool)
    pair_cost = np.asarray(constraints.get("pair_cost"), dtype=float)
    species_meta = list(constraints.get("species_meta") or [])
    if hard_mask.shape != (f.shape[0], f.shape[0]):
        raise ValueError("hard_mask shape mismatch")
    if pair_cost.shape != (f.shape[0], f.shape[0]):
        raise ValueError("pair_cost shape mismatch")

    train_cfg = dict(c.get("train") or {})
    target_ratio = float(train_cfg.get("target_ratio", 0.7))
    min_clusters = int(train_cfg.get("min_clusters", 1))
    min_active_clusters = int(train_cfg.get("min_active_clusters", min_clusters))
    coverage_target = float(train_cfg.get("coverage_target", 0.0))
    coverage_max_clusters = int(train_cfg.get("coverage_max_clusters", f.shape[0]))
    max_cluster_size_ratio = float(train_cfg.get("max_cluster_size_ratio", 1.0))
    n_clusters = int(max(1, round(f.shape[0] * target_ratio)))
    model_cfg = dict(c.get("model") or {})
    model_cfg["n_clusters"] = n_clusters

    model = build_pooling_model(input_dim=f.shape[1], cfg={"model": model_cfg, "train": train_cfg})
    s_prob = infer_assignment(model, graph, f, {"train": train_cfg})

    S = _hard_assign(s_prob)
    S = _repair_hard_violations(S, hard_mask, pair_cost)
    if S.shape[1] != s_prob.shape[1]:
        # Repair can introduce extra singleton clusters to satisfy hard-ban exactly.
        # Keep probability tensor aligned with repaired assignment.
        s_prob = np.asarray(S, dtype=float).copy()

    # Remove empty clusters and re-normalize.
    keep_cols = np.where(np.sum(S, axis=0) > 0.0)[0]
    if keep_cols.size == 0:
        S = np.eye(f.shape[0], dtype=float)
        s_prob = np.eye(f.shape[0], dtype=float)
    else:
        S = S[:, keep_cols]
        s_prob = s_prob[:, keep_cols]
        row_sum = np.sum(s_prob, axis=1, keepdims=True)
        row_sum = np.where(row_sum > 0.0, row_sum, 1.0)
        s_prob = s_prob / row_sum

    S = _enforce_min_clusters(S, min_clusters=max(min_clusters, min_active_clusters))
    species_weights = np.linalg.norm(f, axis=1)
    S, coverage_proxy = _repair_coverage_target(
        S,
        species_weights=species_weights,
        coverage_target=coverage_target,
        coverage_max_clusters=coverage_max_clusters,
    )
    S, max_cluster_ratio_obs = _repair_max_cluster_size(
        S,
        species_weights=species_weights,
        max_cluster_size_ratio=max_cluster_size_ratio,
    )
    S = _enforce_min_clusters(S, min_clusters=max(min_clusters, min_active_clusters))
    if s_prob.shape[1] != S.shape[1]:
        s_prob = np.asarray(S, dtype=float).copy()

    hard_viol = 0
    for cidx in range(S.shape[1]):
        members = np.where(S[:, cidx] > 0.5)[0]
        for i_pos, i in enumerate(members):
            for j in members[i_pos + 1 :]:
                if not hard_mask[i, j]:
                    hard_viol += 1

    loss = pooling_constraint_loss(s_prob, hard_mask, pair_cost, c.get("constraints"))

    cluster_guard_passed = float(max_cluster_ratio_obs) <= float(max_cluster_size_ratio) + 1.0e-12

    return {
        "S": S,
        "S_prob": s_prob,
        "cluster_meta": _cluster_meta(S, species_meta),
        "train_metrics": {
            "constraint_loss": float(loss),
            "hard_ban_violations": int(hard_viol),
            "n_clusters": int(S.shape[1]),
            "n_species": int(S.shape[0]),
            "target_ratio": float(target_ratio),
            "min_clusters": int(max(1, min(min_clusters, S.shape[0]))),
            "min_active_clusters": int(max(1, min(min_active_clusters, S.shape[0]))),
            "coverage_target": float(max(0.0, min(1.0, coverage_target))),
            "coverage_proxy": float(coverage_proxy),
            "max_cluster_size_ratio": float(max_cluster_ratio_obs),
            "max_cluster_size_ratio_limit": float(np.clip(max_cluster_size_ratio, 0.0, 1.0)),
            "cluster_guard_passed": bool(cluster_guard_passed),
        },
        "model_info": {
            "model_type": type(model).__name__,
            "graph_type": str(graph.get("type", "unknown")),
        },
    }
