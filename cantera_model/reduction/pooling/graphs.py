from __future__ import annotations

from typing import Any

import numpy as np


def _normalize(mat: np.ndarray) -> np.ndarray:
    arr = np.asarray(mat, dtype=float)
    if arr.size == 0:
        return arr
    mx = float(np.max(np.abs(arr)))
    if mx <= 0.0:
        return np.zeros_like(arr)
    return arr / mx


def build_species_graph(
    nu: np.ndarray,
    F_bar: np.ndarray | None,
    species_meta: list[dict[str, Any]],
    cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    nu_arr = np.asarray(nu, dtype=float)
    if nu_arr.ndim != 2:
        raise ValueError("nu must be 2-D")
    n_species, n_reactions = nu_arr.shape
    if len(species_meta) != n_species:
        raise ValueError("species_meta length must match nu species dimension")

    adjacency = np.zeros((n_species, n_species), dtype=float)
    # Co-participation in reactions.
    participates = np.abs(nu_arr) > 0.0
    for j in range(n_reactions):
        idx = np.where(participates[:, j])[0]
        for a in idx:
            adjacency[a, idx] += 1.0
    adjacency = _normalize(adjacency)

    if F_bar is not None:
        f_arr = np.asarray(F_bar, dtype=float)
        if f_arr.shape != (n_species, n_species):
            raise ValueError("F_bar shape mismatch")
        adjacency = 0.5 * adjacency + 0.5 * _normalize(np.abs(f_arr))

    np.fill_diagonal(adjacency, 0.0)
    row, col = np.where(adjacency > 0.0)
    edge_index = np.vstack([row, col]).astype(np.int64) if row.size else np.zeros((2, 0), dtype=np.int64)
    edge_weight = adjacency[row, col] if row.size else np.zeros((0,), dtype=float)

    return {
        "type": "species_graph",
        "num_species": int(n_species),
        "num_reactions": int(n_reactions),
        "adjacency": adjacency,
        "edge_index": edge_index,
        "edge_weight": np.asarray(edge_weight, dtype=float),
    }


def build_bipartite_graph(
    nu: np.ndarray,
    rop_stats: np.ndarray | None,
    species_meta: list[dict[str, Any]],
    cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    nu_arr = np.asarray(nu, dtype=float)
    if nu_arr.ndim != 2:
        raise ValueError("nu must be 2-D")
    n_species, n_reactions = nu_arr.shape
    if len(species_meta) != n_species:
        raise ValueError("species_meta length must match nu species dimension")

    if rop_stats is None:
        rop_w = np.ones((n_reactions,), dtype=float)
    else:
        rop_w = np.asarray(rop_stats, dtype=float).reshape(-1)
        if rop_w.shape != (n_reactions,):
            raise ValueError("rop_stats must have shape (Nr,)")
        rop_w = np.abs(rop_w)
        mx = float(np.max(rop_w)) if rop_w.size else 0.0
        if mx > 0.0:
            rop_w = rop_w / mx

    sp_idx: list[int] = []
    rx_idx: list[int] = []
    edge_weight: list[float] = []
    for i in range(n_species):
        for j in np.where(np.abs(nu_arr[i, :]) > 0.0)[0]:
            sp_idx.append(i)
            rx_idx.append(int(j))
            edge_weight.append(float(abs(nu_arr[i, j]) * rop_w[j]))

    if edge_weight:
        ew = np.asarray(edge_weight, dtype=float)
        mx = float(np.max(ew))
        if mx > 0.0:
            ew = ew / mx
    else:
        ew = np.zeros((0,), dtype=float)

    return {
        "type": "bipartite_graph",
        "num_species": int(n_species),
        "num_reactions": int(n_reactions),
        "species_index": np.asarray(sp_idx, dtype=np.int64),
        "reaction_index": np.asarray(rx_idx, dtype=np.int64),
        "edge_weight": ew,
    }
