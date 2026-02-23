from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from cantera_model.types import ReductionMapping


DEFAULT_POLICY: dict[str, Any] = {
    "hard": {"element_overlap_required": True, "phase_mixing_forbidden": False, "surface_site_family_strict": False},
    "soft": {"penalty": {"phase": 0.8, "charge": 0.8, "radical": 0.4, "role": 0.3}},
    "weights": {"elem": 1.0, "frag": 0.8, "flux": 1.2, "role": 0.5},
    "overlap_method": "jaccard",
    "search": {"min_pair_score": -1.0e9},
}


@dataclass
class _UnionState:
    parent: list[int]
    members: dict[int, set[int]]

    @classmethod
    def create(cls, n: int) -> "_UnionState":
        return cls(parent=list(range(n)), members={i: {i} for i in range(n)})

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def can_merge(self, a: int, b: int, candidate_mask: np.ndarray) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        for i in self.members[ra]:
            for j in self.members[rb]:
                if not bool(candidate_mask[i, j]):
                    return False
        return True

    def merge(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if len(self.members[ra]) < len(self.members[rb]):
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.members[ra].update(self.members.pop(rb))
        return True


def _composition(meta: dict[str, Any]) -> dict[str, float]:
    comp = meta.get("composition") or {}
    if not isinstance(comp, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in comp.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv > 0.0:
            out[str(k)] = fv
    return out


def _elements(meta: dict[str, Any]) -> set[str]:
    return set(_composition(meta).keys())


def _phase_bucket(meta: dict[str, Any]) -> str:
    phase = str(meta.get("phase", "")).strip().lower()
    if not phase:
        return ""
    if "gas" in phase:
        return "gas"
    if any(tok in phase for tok in ("bulk", "solid", "deposit")):
        return "bulk"
    if any(tok in phase for tok in ("surface", "interface", "diamond", "si3n4")):
        return "surface"
    return phase


def _surface_site_family(meta: dict[str, Any]) -> str:
    explicit = str(meta.get("site_family", "")).strip().upper()
    if explicit:
        return explicit
    name = str(meta.get("name", "")).strip()
    if not name:
        return ""

    lower = name.lower()
    prefix = re.match(r"^([tskdn])[_:]", lower)
    if prefix:
        return prefix.group(1).upper()

    suffix = re.search(r"_([tskdn])\(", lower)
    if suffix:
        return suffix.group(1).upper()

    alt = re.search(r"_([tskdn])(?:$|[^a-z0-9])", lower)
    if alt:
        return alt.group(1).upper()
    return ""


def build_phase_site_mask(species_meta: list[dict[str, Any]], policy: dict[str, Any] | None = None) -> np.ndarray:
    pol = dict(policy or {})
    hard = dict(pol.get("hard") or {})
    phase_mixing_forbidden = bool(hard.get("phase_mixing_forbidden", False))
    surface_site_family_strict = bool(hard.get("surface_site_family_strict", False))

    n = len(species_meta)
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, True)
    phase_buckets = [_phase_bucket(m) for m in species_meta]
    site_families = [_surface_site_family(m) for m in species_meta]

    for i in range(n):
        for j in range(i + 1, n):
            allow = True
            if phase_mixing_forbidden and phase_buckets[i] and phase_buckets[j]:
                allow = allow and (phase_buckets[i] == phase_buckets[j])
            if surface_site_family_strict and phase_buckets[i] == "surface" and phase_buckets[j] == "surface":
                fam_i = site_families[i]
                fam_j = site_families[j]
                if fam_i and fam_j:
                    allow = allow and (fam_i == fam_j)
            mask[i, j] = allow
            mask[j, i] = allow
    return mask


def build_candidate_mask(species_meta: list[dict[str, Any]], policy: dict[str, Any] | None = None) -> np.ndarray:
    n = len(species_meta)
    mask = np.zeros((n, n), dtype=bool)
    np.fill_diagonal(mask, True)
    phase_site = build_phase_site_mask(species_meta, policy)

    for i in range(n):
        ei = _elements(species_meta[i])
        if not ei:
            continue
        for j in range(i + 1, n):
            ej = _elements(species_meta[j])
            if not ej:
                continue
            allow = len(ei & ej) > 0
            allow = allow and bool(phase_site[i, j])
            mask[i, j] = allow
            mask[j, i] = allow
    return mask


def elem_overlap_score(comp_i: dict[str, float], comp_j: dict[str, float], method: str = "jaccard") -> float:
    keys_i = {k for k, v in comp_i.items() if v > 0}
    keys_j = {k for k, v in comp_j.items() if v > 0}
    if not keys_i or not keys_j:
        return 0.0

    if method == "minmax":
        common = keys_i & keys_j
        if not common:
            return 0.0
        num = sum(min(comp_i.get(k, 0.0), comp_j.get(k, 0.0)) for k in common)
        den = sum(max(comp_i.get(k, 0.0), comp_j.get(k, 0.0)) for k in (keys_i | keys_j))
        return float(num / den) if den > 0 else 0.0

    inter = keys_i & keys_j
    union = keys_i | keys_j
    return float(len(inter) / len(union)) if union else 0.0


def _formula_token(name: str) -> str:
    chars = [c for c in name if c.isalpha()]
    return "".join(chars).upper()


def fragment_affinity(meta_i: dict[str, Any], meta_j: dict[str, Any]) -> float:
    comp_i = _composition(meta_i)
    comp_j = _composition(meta_j)
    if not comp_i or not comp_j:
        return 0.0

    shared = set(comp_i) & set(comp_j)
    if not shared:
        return 0.0

    # Weighted Dice on atomic counts; highlights fragment-like relation.
    num = 2.0 * sum(min(comp_i[k], comp_j[k]) for k in shared)
    den = sum(comp_i.values()) + sum(comp_j.values())
    comp_score = float(num / den) if den > 0 else 0.0

    token_i = _formula_token(str(meta_i.get("name", "")))
    token_j = _formula_token(str(meta_j.get("name", "")))
    name_score = 0.0
    if token_i and token_j:
        if token_i in token_j or token_j in token_i:
            name_score = 1.0
        else:
            set_i = set(token_i)
            set_j = set(token_j)
            name_score = len(set_i & set_j) / max(len(set_i | set_j), 1)

    return float(np.clip(0.7 * comp_score + 0.3 * name_score, 0.0, 1.0))


def _normalized_flux(F_bar: np.ndarray | None) -> np.ndarray | None:
    if F_bar is None:
        return None
    arr = np.asarray(F_bar, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("F_bar must be square")
    max_val = float(np.max(np.abs(arr))) if arr.size else 0.0
    if max_val > 0.0:
        arr = arr / max_val
    return arr


def _role_similarity(meta_i: dict[str, Any], meta_j: dict[str, Any]) -> float:
    ri = str(meta_i.get("role", "")).strip()
    rj = str(meta_j.get("role", "")).strip()
    if not ri and not rj:
        return 0.5
    return 1.0 if ri and ri == rj else 0.0


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def _lumpability_similarity(i: int, j: int, flux: np.ndarray | None) -> float:
    if flux is None:
        return 0.0
    row_sim = _cosine_similarity(np.abs(flux[i, :]), np.abs(flux[j, :]))
    col_sim = _cosine_similarity(np.abs(flux[:, i]), np.abs(flux[:, j]))
    return float(0.5 * (row_sim + col_sim))


def _soft_penalty(
    meta_i: dict[str, Any],
    meta_j: dict[str, Any],
    policy: dict[str, Any],
    lumpability_similarity: float | None = None,
) -> float:
    pen = (policy.get("soft") or {}).get("penalty") or {}
    soft_cfg = dict(policy.get("soft") or {})
    out = 0.0

    if str(meta_i.get("phase", "")) != str(meta_j.get("phase", "")):
        out += float(pen.get("phase", 0.0))
    if str(meta_i.get("charge", "")) != str(meta_j.get("charge", "")):
        out += float(pen.get("charge", 0.0))
    if bool(meta_i.get("radical", False)) != bool(meta_j.get("radical", False)):
        out += float(pen.get("radical", 0.0))
    if str(meta_i.get("role", "")) != str(meta_j.get("role", "")):
        out += float(pen.get("role", 0.0))
    if lumpability_similarity is not None:
        lump_w = float(pen.get("lumpability", soft_cfg.get("lumpability_weight", 0.0)))
        out += lump_w * float(max(0.0, 1.0 - lumpability_similarity))

    return out


def build_similarity(
    species_meta: list[dict[str, Any]],
    F_bar: np.ndarray | None,
    policy: dict[str, Any] | None = None,
) -> np.ndarray:
    n = len(species_meta)
    pol = dict(DEFAULT_POLICY)
    if policy:
        # shallow merge is sufficient for expected config layout
        for key, value in policy.items():
            if isinstance(value, dict) and isinstance(pol.get(key), dict):
                merged = dict(pol[key])
                merged.update(value)
                pol[key] = merged
            else:
                pol[key] = value

    weights = pol.get("weights") or {}
    w_elem = float(weights.get("elem", 1.0))
    w_frag = float(weights.get("frag", 1.0))
    w_flux = float(weights.get("flux", 1.0))
    w_role = float(weights.get("role", 0.0))
    overlap_method = str(pol.get("overlap_method", "jaccard"))

    candidate = build_candidate_mask(species_meta, pol)
    flux = _normalized_flux(F_bar)

    sim = np.full((n, n), -1.0e9, dtype=float)
    np.fill_diagonal(sim, 1.0)

    for i in range(n):
        comp_i = _composition(species_meta[i])
        for j in range(i + 1, n):
            if not candidate[i, j]:
                continue

            comp_j = _composition(species_meta[j])
            elem = elem_overlap_score(comp_i, comp_j, method=overlap_method)
            frag = fragment_affinity(species_meta[i], species_meta[j])
            role = _role_similarity(species_meta[i], species_meta[j])

            flux_term = 0.0
            if flux is not None:
                flux_term = max(0.0, float(flux[i, j]) + float(flux[j, i]))
            lump_sim = _lumpability_similarity(i, j, flux)

            penalty = _soft_penalty(species_meta[i], species_meta[j], pol, lumpability_similarity=lump_sim)
            score = w_elem * elem + w_frag * frag + w_flux * flux_term + w_role * role - penalty
            sim[i, j] = score
            sim[j, i] = score

    return sim


def _policy_with_scale(policy: dict[str, Any], penalty_scale: float) -> dict[str, Any]:
    out = dict(policy)
    soft = dict((out.get("soft") or {}))
    penalties = dict((soft.get("penalty") or {}))
    for k, v in penalties.items():
        penalties[k] = float(v) * float(penalty_scale)
    soft["penalty"] = penalties
    out["soft"] = soft
    return out


def _cluster_indices(state: _UnionState) -> list[list[int]]:
    roots = sorted(state.members.keys())
    return [sorted(state.members[r]) for r in roots]


def fit_merge_mapping(
    species_meta: list[dict[str, Any]],
    F_bar: np.ndarray | None,
    *,
    target_ratio: float,
    policy: dict[str, Any] | None = None,
    penalty_scale: float = 1.0,
) -> ReductionMapping:
    if not 0.0 < target_ratio <= 1.0:
        raise ValueError("target_ratio must be in (0, 1]")

    pol = DEFAULT_POLICY if policy is None else policy
    pol = _policy_with_scale(pol, penalty_scale)

    n = len(species_meta)
    target_clusters = max(1, int(round(n * target_ratio)))
    candidate = build_candidate_mask(species_meta, pol)
    sim = build_similarity(species_meta, F_bar, pol)
    min_pair_score = float(((pol.get("search") or {}).get("min_pair_score", -1.0e9)))

    state = _UnionState.create(n)

    pairs: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if candidate[i, j]:
                pairs.append((float(sim[i, j]), i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])

    def cluster_count() -> int:
        return len(state.members)

    for score, i, j in pairs:
        if cluster_count() <= target_clusters:
            break
        if score < min_pair_score:
            continue
        if not state.can_merge(i, j, candidate):
            continue
        state.merge(i, j)

    if cluster_count() > target_clusters:
        for score, i, j in pairs:
            if cluster_count() <= target_clusters:
                break
            if not state.can_merge(i, j, candidate):
                continue
            state.merge(i, j)

    clusters = _cluster_indices(state)
    n_clusters = len(clusters)
    S = np.zeros((n, n_clusters), dtype=float)
    pool_meta: list[dict[str, Any]] = []
    for c_idx, members in enumerate(clusters):
        for m in members:
            S[m, c_idx] = 1.0
        names = [str(species_meta[m].get("name", f"sp{m}")) for m in members]
        elem_union: set[str] = set()
        for m in members:
            elem_union |= _elements(species_meta[m])
        pool_meta.append({"cluster_id": c_idx, "members": names, "elements": sorted(elem_union)})

    hard_ban_violations = 0
    for members in clusters:
        for i_idx, i in enumerate(members):
            for j in members[i_idx + 1 :]:
                if not candidate[i, j]:
                    hard_ban_violations += 1

    return ReductionMapping(
        S=S,
        pool_meta=pool_meta,
        keep_reactions=None,
        meta={
            "target_ratio": float(target_ratio),
            "achieved_ratio": float(n_clusters) / float(max(n, 1)),
            "hard_ban_violations": int(hard_ban_violations),
            "candidate_pairs": int(np.sum(np.triu(candidate, 1))),
        },
    )
