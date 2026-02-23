from __future__ import annotations

from typing import Any

import numpy as np

from cantera_model.reduction.merge_free import elem_overlap_score, fragment_affinity


def _composition(meta: dict[str, Any]) -> dict[str, float]:
    comp = meta.get("composition") or {}
    out: dict[str, float] = {}
    if not isinstance(comp, dict):
        return out
    for key, value in comp.items():
        try:
            fv = float(value)
        except (TypeError, ValueError):
            continue
        if fv > 0.0:
            out[str(key)] = fv
    return out


def _cluster_species_indices(S: np.ndarray) -> list[np.ndarray]:
    return [np.where(S[:, k] > 1.0e-12)[0] for k in range(S.shape[1])]


def _cluster_meta(species_meta: list[dict[str, Any]], S: np.ndarray) -> list[dict[str, Any]]:
    cluster_meta: list[dict[str, Any]] = []
    for k, idxs in enumerate(_cluster_species_indices(S)):
        names: list[str] = []
        comp_sum: dict[str, float] = {}
        roles: set[str] = set()
        for i in idxs:
            sp = dict(species_meta[int(i)])
            names.append(str(sp.get("name", f"sp{i}")))
            roles.add(str(sp.get("role", "")).strip())
            for elem, count in _composition(sp).items():
                comp_sum[elem] = comp_sum.get(elem, 0.0) + float(count)
        cluster_meta.append(
            {
                "cluster_id": int(k),
                "name": "+".join(names) if names else f"cluster_{k}",
                "members": names,
                "composition": comp_sum,
                "elements": sorted(comp_sum.keys()),
                "role": "|".join(sorted(r for r in roles if r)),
            }
        )
    return cluster_meta


def _cluster_flux(F_bar: np.ndarray | None, S: np.ndarray) -> np.ndarray:
    n_cluster = S.shape[1]
    if F_bar is None:
        return np.zeros((n_cluster, n_cluster), dtype=float)
    arr = np.asarray(F_bar, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("F_bar must be square")
    if arr.shape[0] != S.shape[0]:
        raise ValueError("F_bar species dimension mismatch with S")
    out = np.abs(S.T @ arr @ S)
    if out.size:
        np.fill_diagonal(out, 0.0)
        max_val = float(np.max(out))
        if max_val > 0.0:
            out = out / max_val
    return out


def _role_similarity(meta_i: dict[str, Any], meta_j: dict[str, Any]) -> float:
    ri = str(meta_i.get("role", "")).strip()
    rj = str(meta_j.get("role", "")).strip()
    if not ri and not rj:
        return 0.5
    return 1.0 if (ri and rj and ri == rj) else 0.0


def backfill_candidates_by_importance(
    selected: list[tuple[float, np.ndarray, dict[str, Any]]],
    reserve: list[tuple[float, np.ndarray, dict[str, Any]]],
    *,
    min_candidates_floor: int,
    max_candidates: int,
) -> list[tuple[float, np.ndarray, dict[str, Any]]]:
    floor = int(max(0, min_candidates_floor))
    if floor <= 0:
        return selected[: max(0, max_candidates)]

    cap = max(0, int(max_candidates))
    if cap == 0:
        return []
    target = min(cap, floor)
    if len(selected) >= target:
        return selected[:cap]

    out = list(selected)
    used = {int(item[2].get("_idx", -1)) for item in out}
    for score, nu_vec, meta in reserve:
        idx = int(meta.get("_idx", -1))
        if idx in used:
            continue
        out.append((score, nu_vec, meta))
        used.add(idx)
        if len(out) >= target:
            break
    out.sort(key=lambda x: float(x[0]), reverse=True)
    return out[:cap]


def _active_clusters_from_candidates(
    candidates: list[tuple[float, np.ndarray, dict[str, Any]]],
    n_clusters: int,
) -> np.ndarray:
    if n_clusters <= 0 or not candidates:
        return np.zeros((max(0, n_clusters),), dtype=bool)
    mask = np.zeros((n_clusters,), dtype=bool)
    for _, nu_vec, _ in candidates:
        arr = np.asarray(nu_vec, dtype=float)
        rows = min(n_clusters, int(arr.shape[0]))
        if rows > 0:
            mask[:rows] |= np.abs(arr[:rows]) > 1.0e-12
    return mask


def backfill_candidates_by_cluster_coverage(
    selected: list[tuple[float, np.ndarray, dict[str, Any]]],
    reserve: list[tuple[float, np.ndarray, dict[str, Any]]],
    *,
    min_candidates_floor: int,
    max_candidates: int,
    n_clusters: int,
    min_active_clusters: int = 0,
) -> list[tuple[float, np.ndarray, dict[str, Any]]]:
    cap = max(0, int(max_candidates))
    if cap <= 0:
        return []

    floor = max(0, int(min_candidates_floor))
    target = min(cap, floor)
    target_active = max(0, min(int(min_active_clusters), int(n_clusters)))

    out = list(selected[:cap])
    used = {int(item[2].get("_idx", -1)) for item in out}

    # First satisfy minimum candidate count.
    if len(out) < target:
        for score, nu_vec, meta in reserve:
            idx = int(meta.get("_idx", -1))
            if idx in used:
                continue
            out.append((score, nu_vec, meta))
            used.add(idx)
            if len(out) >= target:
                break

    # Then greedily improve active-cluster coverage.
    if target_active > 0 and len(out) < cap:
        active_mask = _active_clusters_from_candidates(out, n_clusters)
        while int(np.sum(active_mask)) < target_active and len(out) < cap:
            best: tuple[float, np.ndarray, dict[str, Any]] | None = None
            best_gain = -1
            best_score = -float("inf")
            for cand in reserve:
                meta = cand[2]
                idx = int(meta.get("_idx", -1))
                if idx in used:
                    continue
                nu_vec = np.asarray(cand[1], dtype=float)
                rows = min(n_clusters, int(nu_vec.shape[0]))
                touched = np.zeros((n_clusters,), dtype=bool)
                if rows > 0:
                    touched[:rows] = np.abs(nu_vec[:rows]) > 1.0e-12
                gain = int(np.sum(np.logical_and(touched, ~active_mask)))
                score = float(cand[0])
                if gain > best_gain or (gain == best_gain and score > best_score):
                    best = cand
                    best_gain = gain
                    best_score = score
            if best is None:
                break
            out.append(best)
            used.add(int(best[2].get("_idx", -1)))
            active_mask = _active_clusters_from_candidates(out, n_clusters)

    out.sort(key=lambda x: float(x[0]), reverse=True)
    return out[:cap]


def generate_overall_candidates(
    nu: np.ndarray,
    F_bar: np.ndarray | None,
    S: np.ndarray,
    species_meta: list[dict[str, Any]],
    policy: dict[str, Any] | None,
) -> dict[str, Any]:
    nu_arr = np.asarray(nu, dtype=float)
    S_arr = np.asarray(S, dtype=float)

    if nu_arr.ndim != 2:
        raise ValueError("nu must be 2-D")
    if S_arr.ndim != 2:
        raise ValueError("S must be 2-D")
    if nu_arr.shape[0] != S_arr.shape[0]:
        raise ValueError("nu and S species dimension mismatch")
    if len(species_meta) != S_arr.shape[0]:
        raise ValueError("species_meta length mismatch with S rows")

    pol = dict(policy or {})
    hard = dict(pol.get("hard") or {})
    cand_cfg = dict(pol.get("candidate") or {})
    weights = dict(pol.get("weights") or {})
    overlap_method = str(pol.get("overlap_method", "jaccard"))

    hard_overlap = bool(hard.get("element_overlap_required", True))
    max_candidates = int(cand_cfg.get("max_candidates", 256))
    min_candidates_floor = int(cand_cfg.get("min_candidates_floor", 0))
    min_flux_quantile = float(cand_cfg.get("min_flux_quantile", 0.70))
    min_flux_quantile = float(np.clip(min_flux_quantile, 0.0, 1.0))
    backfill_by_uncovered = bool(cand_cfg.get("backfill_by_uncovered_clusters", False))
    min_active_clusters = int(cand_cfg.get("min_active_clusters", 0))

    w_elem = float(weights.get("elem", 1.0))
    w_frag = float(weights.get("frag", 1.0))
    w_flux = float(weights.get("flux", 1.0))
    w_role = float(weights.get("role", 0.2))

    cluster_meta = _cluster_meta(species_meta, S_arr)
    n_cluster = len(cluster_meta)
    flux = _cluster_flux(F_bar, S_arr)

    flux_vals = flux[flux > 0.0]
    flux_threshold = float(np.quantile(flux_vals, min_flux_quantile)) if flux_vals.size else 0.0

    candidates_all: list[tuple[float, np.ndarray, dict[str, Any]]] = []
    candidates_flux_pass: list[tuple[float, np.ndarray, dict[str, Any]]] = []
    hard_ban_violations = 0

    pair_idx = 0
    for i in range(n_cluster):
        meta_i = cluster_meta[i]
        comp_i = dict(meta_i.get("composition") or {})
        elems_i = set(meta_i.get("elements") or [])
        for j in range(n_cluster):
            if i == j:
                continue
            meta_j = cluster_meta[j]
            comp_j = dict(meta_j.get("composition") or {})
            elems_j = set(meta_j.get("elements") or [])
            shared = sorted(elems_i & elems_j)

            if hard_overlap and not shared:
                hard_ban_violations += 1
                continue

            flux_score = float(flux[i, j])
            elem_score = elem_overlap_score(comp_i, comp_j, method=overlap_method)
            frag_score = fragment_affinity(meta_i, meta_j)
            role_score = _role_similarity(meta_i, meta_j)
            score = (
                w_elem * float(elem_score)
                + w_frag * float(frag_score)
                + w_flux * flux_score
                + w_role * role_score
            )

            nu_vec = np.zeros((n_cluster,), dtype=float)
            nu_vec[i] = -1.0
            nu_vec[j] = 1.0
            meta = {
                "reactant_cluster": int(i),
                "product_cluster": int(j),
                "reactant_name": str(meta_i.get("name", f"cluster_{i}")),
                "product_name": str(meta_j.get("name", f"cluster_{j}")),
                "shared_elements": shared,
                "flux_score": flux_score,
                "elem_score": float(elem_score),
                "fragment_score": float(frag_score),
                "role_score": float(role_score),
                "score": float(score),
                "_idx": pair_idx,
            }
            tup = (float(score), nu_vec, meta)
            candidates_all.append(tup)
            if (not flux_vals.size) or flux_score >= flux_threshold:
                candidates_flux_pass.append(tup)
            pair_idx += 1

    candidates_flux_pass.sort(key=lambda x: x[0], reverse=True)
    candidates_all.sort(key=lambda x: x[0], reverse=True)
    if max_candidates > 0:
        candidates_flux_pass = candidates_flux_pass[:max_candidates]
        candidates = backfill_candidates_by_importance(
            candidates_flux_pass,
            candidates_all,
            min_candidates_floor=min_candidates_floor,
            max_candidates=max_candidates,
        )
        if backfill_by_uncovered:
            candidates = backfill_candidates_by_cluster_coverage(
                candidates,
                candidates_all,
                min_candidates_floor=min_candidates_floor,
                max_candidates=max_candidates,
                n_clusters=n_cluster,
                min_active_clusters=min_active_clusters,
            )
    else:
        candidates = []

    if not candidates:
        nu_out = np.zeros((n_cluster, 0), dtype=float)
        meta_out: list[dict[str, Any]] = []
    else:
        nu_out = np.stack([c[1] for c in candidates], axis=1)
        meta_out = [{k: v for k, v in c[2].items() if k != "_idx"} for c in candidates]

    return {
        "nu_overall_candidates": nu_out,
        "candidate_meta": meta_out,
        "cluster_meta": cluster_meta,
        "flux_threshold": flux_threshold,
        "min_candidates_floor": int(min_candidates_floor),
        "min_active_clusters": int(max(0, min_active_clusters)),
        "backfill_by_uncovered_clusters": bool(backfill_by_uncovered),
        "candidates_pre_filter": int(len(candidates_all)),
        "candidates_flux_pass": int(len(candidates_flux_pass)),
        "candidates_final": int(nu_out.shape[1]),
        "hard_ban_violations": int(hard_ban_violations),
    }
