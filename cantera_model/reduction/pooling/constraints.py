from __future__ import annotations

import re
from typing import Any

import numpy as np


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


def build_surface_site_mask(species_meta: list[dict[str, Any]], cfg: dict[str, Any] | None = None) -> np.ndarray:
    hard = dict((dict(cfg or {})).get("hard") or {})
    phase_mixing_forbidden = bool(hard.get("phase_mixing_forbidden", False))
    surface_site_family_strict = bool(hard.get("surface_site_family_strict", False))

    n = len(species_meta)
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, True)
    phase_buckets = [_phase_bucket(m) for m in species_meta]
    families = [_surface_site_family(m) for m in species_meta]

    for i in range(n):
        for j in range(i + 1, n):
            allow = True
            if phase_mixing_forbidden and phase_buckets[i] and phase_buckets[j]:
                allow = allow and (phase_buckets[i] == phase_buckets[j])
            if surface_site_family_strict and phase_buckets[i] == "surface" and phase_buckets[j] == "surface":
                fi = families[i]
                fj = families[j]
                if fi and fj:
                    allow = allow and (fi == fj)
            mask[i, j] = allow
            mask[j, i] = allow
    return mask


def build_hard_mask(species_meta: list[dict[str, Any]], cfg: dict[str, Any] | None = None) -> np.ndarray:
    c = dict(cfg or {})
    hard = dict(c.get("hard") or {})
    require_overlap = bool(hard.get("element_overlap_required", True))
    require_same_phase = bool(hard.get("same_phase_required", False))

    n = len(species_meta)
    mask = np.zeros((n, n), dtype=bool)
    np.fill_diagonal(mask, True)
    phase_site_mask = build_surface_site_mask(species_meta, cfg)

    for i in range(n):
        ei = _elements(species_meta[i])
        for j in range(i + 1, n):
            ej = _elements(species_meta[j])
            allow = True
            if require_overlap:
                allow = allow and bool(ei & ej)
            if require_same_phase:
                allow = allow and (str(species_meta[i].get("phase", "")) == str(species_meta[j].get("phase", "")))
            allow = allow and bool(phase_site_mask[i, j])
            mask[i, j] = allow
            mask[j, i] = allow
    return mask


def build_pairwise_cost(species_meta: list[dict[str, Any]], cfg: dict[str, Any] | None = None) -> np.ndarray:
    c = dict(cfg or {})
    pen = dict((c.get("soft") or {}).get("penalty") or {})
    w_phase = float(pen.get("phase", 0.8))
    w_charge = float(pen.get("charge", 0.8))
    w_radical = float(pen.get("radical", 0.4))
    w_role = float(pen.get("role", 0.3))

    n = len(species_meta)
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            cost = 0.0
            if str(species_meta[i].get("phase", "")) != str(species_meta[j].get("phase", "")):
                cost += w_phase
            if str(species_meta[i].get("charge", 0)) != str(species_meta[j].get("charge", 0)):
                cost += w_charge
            if bool(species_meta[i].get("radical", False)) != bool(species_meta[j].get("radical", False)):
                cost += w_radical
            if str(species_meta[i].get("role", "")) != str(species_meta[j].get("role", "")):
                cost += w_role
            out[i, j] = cost
            out[j, i] = cost
    return out


def pooling_constraint_loss(
    S_prob: np.ndarray,
    hard_mask: np.ndarray,
    pair_cost: np.ndarray,
    cfg: dict[str, Any] | None = None,
) -> float:
    s = np.asarray(S_prob, dtype=float)
    hard = np.asarray(hard_mask, dtype=bool)
    cost = np.asarray(pair_cost, dtype=float)

    if s.ndim != 2:
        raise ValueError("S_prob must be 2-D")
    n = s.shape[0]
    if hard.shape != (n, n) or cost.shape != (n, n):
        raise ValueError("hard_mask/pair_cost shape mismatch")

    col_sum = np.sum(s, axis=1, keepdims=True)
    col_sum = np.where(col_sum > 0.0, col_sum, 1.0)
    s = np.maximum(s, 0.0) / col_sum

    co_assign = s @ s.T
    alpha = float((cfg or {}).get("hard_weight", 10.0))
    beta = float((cfg or {}).get("soft_weight", 1.0))

    violation = np.sum(co_assign[~hard])
    soft = np.sum(co_assign * cost)
    norm = float(max(n * n, 1))
    return float((alpha * violation + beta * soft) / norm)
