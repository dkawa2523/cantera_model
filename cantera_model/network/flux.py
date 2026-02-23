from __future__ import annotations

import numpy as np


def _as_dt_array(dt: float | np.ndarray | None, n_t: int) -> np.ndarray:
    if dt is None:
        return np.ones(n_t, dtype=float)
    if np.isscalar(dt):
        return np.full(n_t, float(dt), dtype=float)
    arr = np.asarray(dt, dtype=float)
    if arr.shape != (n_t,):
        raise ValueError("dt array must have shape (T,)")
    return arr


def reaction_importance(rop: np.ndarray, dt: float | np.ndarray | None = None) -> np.ndarray:
    rop_arr = np.asarray(rop, dtype=float)
    if rop_arr.ndim != 2:
        raise ValueError("rop must be 2-D with shape (T, Nr)")
    dt_arr = _as_dt_array(dt, rop_arr.shape[0])
    return np.sum(np.abs(rop_arr) * dt_arr[:, None], axis=0)


def build_flux(
    nu: np.ndarray,
    rop: np.ndarray,
    dt: float | np.ndarray | None = None,
    *,
    normalize: bool = True,
) -> np.ndarray:
    dense_nu = nu.toarray() if hasattr(nu, "toarray") else np.asarray(nu, dtype=float)
    rop_arr = np.asarray(rop, dtype=float)
    if dense_nu.ndim != 2:
        raise ValueError("nu must be 2-D with shape (Ns, Nr)")
    if rop_arr.ndim != 2:
        raise ValueError("rop must be 2-D with shape (T, Nr)")
    if dense_nu.shape[1] != rop_arr.shape[1]:
        raise ValueError("nu reactions dimension must match rop")

    n_species, n_reactions = dense_nu.shape
    dt_arr = _as_dt_array(dt, rop_arr.shape[0])
    consume = np.clip(-dense_nu, 0.0, None)
    produce = np.clip(dense_nu, 0.0, None)

    F_bar = np.zeros((n_species, n_species), dtype=float)
    for j in range(n_reactions):
        cons = consume[:, j]
        prod = produce[:, j]
        denom = cons.sum()
        if denom <= 0.0 or prod.sum() <= 0.0:
            continue
        intensity = float(np.sum(np.abs(rop_arr[:, j]) * dt_arr))
        if intensity <= 0.0:
            continue
        F_bar += (intensity / denom) * np.outer(cons, prod)

    if normalize:
        max_val = float(np.max(F_bar)) if F_bar.size else 0.0
        if max_val > 0.0:
            F_bar = F_bar / max_val

    np.fill_diagonal(F_bar, 0.0)
    return F_bar
