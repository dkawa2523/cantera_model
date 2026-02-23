from __future__ import annotations

from typing import Any

import numpy as np

from cantera_model.reduction.conservation import project_to_conservation
from cantera_model.reduction.learnckpp.rate_model import predict_rates


def _time_features(time: np.ndarray) -> np.ndarray:
    t = np.asarray(time, dtype=float)
    if t.ndim != 1:
        raise ValueError("time must be 1-D")
    if t.size == 0:
        return np.zeros((0, 3), dtype=float)
    t0 = float(t[0])
    span = max(float(t[-1] - t0), 1.0e-12)
    x = (t - t0) / span
    return np.column_stack([np.ones_like(x), x, x * x])


def _adapt_features(features: np.ndarray, expected_dim: int) -> np.ndarray:
    arr = np.asarray(features, dtype=float)
    if arr.ndim != 2:
        raise ValueError("features must be 2-D")
    if arr.shape[1] == expected_dim:
        return arr
    if arr.shape[1] > expected_dim:
        return arr[:, :expected_dim]
    pad = np.zeros((arr.shape[0], expected_dim - arr.shape[1]), dtype=float)
    return np.concatenate([arr, pad], axis=1)


def simulate_reduced(
    y0: np.ndarray,
    time: np.ndarray,
    model_artifact: dict[str, Any],
    nu_overall_sel: np.ndarray,
    proj_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    y0_arr = np.asarray(y0, dtype=float)
    t_arr = np.asarray(time, dtype=float)
    nu_sel = np.asarray(nu_overall_sel, dtype=float)

    if y0_arr.ndim != 1:
        raise ValueError("y0 must be 1-D")
    if t_arr.ndim != 1:
        raise ValueError("time must be 1-D")
    if nu_sel.ndim != 2:
        raise ValueError("nu_overall_sel must be 2-D")
    if nu_sel.shape[0] != y0_arr.shape[0]:
        raise ValueError("nu_overall_sel rows must match y0 size")

    art = dict(model_artifact or {})
    feature_dim = int(art.get("feature_dim", 3))
    if "sim_features" in art:
        features = np.asarray(art["sim_features"], dtype=float)
        if features.shape[0] != t_arr.shape[0]:
            features = _time_features(t_arr)
    else:
        features = _time_features(t_arr)
    features = _adapt_features(features, feature_dim)

    rates = predict_rates(art, features)
    if rates.shape[0] != t_arr.shape[0]:
        raise ValueError("predicted rates rows must match time size")
    if rates.shape[1] != nu_sel.shape[1]:
        # Keep simulation robust if selection/fitting dimensions drift.
        if rates.shape[1] > nu_sel.shape[1]:
            rates = rates[:, : nu_sel.shape[1]]
        else:
            pad = np.zeros((rates.shape[0], nu_sel.shape[1] - rates.shape[1]), dtype=float)
            rates = np.concatenate([rates, pad], axis=1)

    ydot = rates @ nu_sel.T

    y = np.zeros((t_arr.shape[0], y0_arr.shape[0]), dtype=float)
    if t_arr.size == 0:
        return y, ydot
    y[0] = y0_arr
    dt = np.zeros_like(t_arr)
    if t_arr.size > 1:
        dt[1:] = np.maximum(np.diff(t_arr), 0.0)
    for i in range(1, y.shape[0]):
        y[i] = y[i - 1] + ydot[i - 1] * dt[i]

    cfg = dict(proj_cfg or {})
    enabled = bool(cfg.get("enabled", True))
    clip_nonnegative = bool(cfg.get("clip_nonnegative", True))
    if enabled:
        A = np.asarray(cfg.get("A"), dtype=float)
        max_iter = int(cfg.get("max_iter", 4))
        y = np.asarray(
            project_to_conservation(
                y,
                A,
                reference=np.asarray(cfg.get("reference", y0_arr), dtype=float),
                clip_nonnegative=clip_nonnegative,
                max_iter=max_iter,
            ),
            dtype=float,
        )
    elif clip_nonnegative:
        y = np.maximum(y, 0.0)

    return y, ydot
