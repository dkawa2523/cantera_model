from __future__ import annotations

from typing import Any

import numpy as np


def _composition(meta: dict[str, Any]) -> dict[str, float]:
    comp = meta.get("composition") or {}
    out: dict[str, float] = {}
    if not isinstance(comp, dict):
        return out
    for k, v in comp.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv > 0.0:
            out[str(k)] = fv
    return out


def extract_species_features(
    species_meta: list[dict[str, Any]],
    X: np.ndarray,
    wdot: np.ndarray,
    phase_labels: list[str] | None,
    cfg: dict[str, Any] | None,
) -> np.ndarray:
    x_arr = np.asarray(X, dtype=float)
    w_arr = np.asarray(wdot, dtype=float)
    if x_arr.ndim != 2 or w_arr.ndim != 2:
        raise ValueError("X and wdot must be 2-D")
    if x_arr.shape != w_arr.shape:
        raise ValueError("X and wdot shape mismatch")
    n_steps, n_species = x_arr.shape
    if len(species_meta) != n_species:
        raise ValueError("species_meta length must match X species dimension")

    all_elements = sorted({e for sp in species_meta for e in _composition(sp).keys()})
    elem_dim = len(all_elements)
    feat = np.zeros((n_species, elem_dim + 9), dtype=float)

    mean_x = np.mean(x_arr, axis=0) if n_steps > 0 else np.zeros((n_species,), dtype=float)
    max_x = np.max(x_arr, axis=0) if n_steps > 0 else np.zeros((n_species,), dtype=float)
    mean_abs_w = np.mean(np.abs(w_arr), axis=0) if n_steps > 0 else np.zeros((n_species,), dtype=float)
    max_abs_w = np.max(np.abs(w_arr), axis=0) if n_steps > 0 else np.zeros((n_species,), dtype=float)

    phases = sorted({str(m.get("phase", "")) for m in species_meta})
    phase_to_idx = {p: i for i, p in enumerate(phases)}

    for i, sp in enumerate(species_meta):
        comp = _composition(sp)
        for e_idx, elem in enumerate(all_elements):
            feat[i, e_idx] = float(comp.get(elem, 0.0))

        base = elem_dim
        feat[i, base + 0] = float(mean_x[i])
        feat[i, base + 1] = float(max_x[i])
        feat[i, base + 2] = float(mean_abs_w[i])
        feat[i, base + 3] = float(max_abs_w[i])
        feat[i, base + 4] = float(bool(sp.get("radical", False)))
        feat[i, base + 5] = float(sp.get("charge", 0.0) or 0.0)
        feat[i, base + 6] = float(sum(comp.values()))
        phase = str(sp.get("phase", ""))
        feat[i, base + 7] = float(phase_to_idx.get(phase, -1))
        role = str(sp.get("role", ""))
        feat[i, base + 8] = float((sum(ord(c) for c in role) % 101) / 100.0)

    # Normalize columns for stable downstream optimization.
    col_scale = np.max(np.abs(feat), axis=0, keepdims=True)
    col_scale = np.where(col_scale > 0.0, col_scale, 1.0)
    return feat / col_scale


def extract_reaction_features(
    nu: np.ndarray,
    rop: np.ndarray,
    cfg: dict[str, Any] | None,
) -> np.ndarray:
    nu_arr = np.asarray(nu, dtype=float)
    rop_arr = np.asarray(rop, dtype=float)
    if nu_arr.ndim != 2 or rop_arr.ndim != 2:
        raise ValueError("nu and rop must be 2-D")
    n_species, n_reactions = nu_arr.shape
    if rop_arr.shape[1] != n_reactions:
        raise ValueError("rop reaction dimension mismatch with nu")

    mean_abs_rop = np.mean(np.abs(rop_arr), axis=0) if rop_arr.shape[0] > 0 else np.zeros((n_reactions,), dtype=float)
    max_abs_rop = np.max(np.abs(rop_arr), axis=0) if rop_arr.shape[0] > 0 else np.zeros((n_reactions,), dtype=float)
    reactant_count = np.sum(nu_arr < 0.0, axis=0)
    product_count = np.sum(nu_arr > 0.0, axis=0)
    stoich_l1 = np.sum(np.abs(nu_arr), axis=0)
    stoich_l2 = np.sqrt(np.sum(nu_arr * nu_arr, axis=0))

    out = np.column_stack(
        [
            mean_abs_rop,
            max_abs_rop,
            reactant_count,
            product_count,
            stoich_l1,
            stoich_l2,
        ]
    )
    scale = np.max(np.abs(out), axis=0, keepdims=True)
    scale = np.where(scale > 0.0, scale, 1.0)
    return out / scale
