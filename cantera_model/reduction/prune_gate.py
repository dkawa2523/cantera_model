from __future__ import annotations

from typing import Any

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _to_dense(arr: np.ndarray) -> np.ndarray:
    return arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr, dtype=float)


def train_prune_gate(
    nu: np.ndarray,
    rop: np.ndarray,
    wdot: np.ndarray,
    *,
    lambda_l0: float = 1.0e-3,
    lr: float = 0.5,
    max_steps: int = 300,
    init_temp: float = 2.0,
    final_temp: float = 0.5,
    threshold: float = 0.5,
    target_keep_ratio: float | None = None,
    enforce_target_exact: bool = True,
    min_keep_count: int | None = None,
    init_importance: np.ndarray | None = None,
    seed: int = 0,
    fallback_quantile: float = 0.2,
    return_details: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    dense_nu = _to_dense(nu)
    rop_arr = np.asarray(rop, dtype=float)
    wdot_arr = np.asarray(wdot, dtype=float)

    if dense_nu.ndim != 2:
        raise ValueError("nu must be 2-D")
    if rop_arr.ndim != 2 or wdot_arr.ndim != 2:
        raise ValueError("rop and wdot must be 2-D")
    if rop_arr.shape[1] != dense_nu.shape[1]:
        raise ValueError("reaction dimension mismatch between rop and nu")
    if wdot_arr.shape[1] != dense_nu.shape[0]:
        raise ValueError("species dimension mismatch between wdot and nu")

    n_reactions = dense_nu.shape[1]
    rng = np.random.default_rng(seed)

    if init_importance is not None:
        imp = np.asarray(init_importance, dtype=float)
        if imp.shape != (n_reactions,):
            raise ValueError("init_importance must have shape (Nr,)")
        imp = np.maximum(imp, 0.0)
        if float(np.max(imp)) > 0.0:
            probs0 = imp / float(np.max(imp))
        else:
            probs0 = np.full(n_reactions, 0.5, dtype=float)
        probs0 = np.clip(probs0, 1.0e-4, 1.0 - 1.0e-4)
        logits = np.log(probs0 / (1.0 - probs0)) + rng.normal(0.0, 0.02, size=n_reactions)
    else:
        logits = rng.normal(0.0, 0.05, size=n_reactions)

    T, Ns = wdot_arr.shape
    temps = np.linspace(init_temp, final_temp, num=max_steps)

    status = "ok"
    for temp in temps:
        z = _sigmoid(logits / max(temp, 1.0e-6))
        pred = (rop_arr * z[None, :]) @ dense_nu.T
        err = pred - wdot_arr

        err_nu = err @ dense_nu
        d_mse_dz = (2.0 / (T * Ns)) * np.sum(rop_arr * err_nu, axis=0)
        d_loss_dz = d_mse_dz + float(lambda_l0)
        d_z_d_logits = (z * (1.0 - z)) / max(temp, 1.0e-6)
        grad = d_loss_dz * d_z_d_logits

        logits -= lr * np.clip(grad, -10.0, 10.0)

        if not np.all(np.isfinite(logits)):
            status = "fallback_non_finite"
            break

    if status != "ok":
        importance = np.sum(np.abs(rop_arr), axis=0)
        cutoff = np.quantile(importance, fallback_quantile)
        keep = importance >= cutoff
        probs = importance / (float(np.max(importance)) + 1.0e-12)
    else:
        probs = _sigmoid(logits / max(final_temp, 1.0e-6))
        keep = probs >= float(threshold)
        if not np.any(keep):
            status = "fallback_all_pruned"
            importance = np.sum(np.abs(rop_arr), axis=0)
            keep[int(np.argmax(importance))] = True

    if target_keep_ratio is not None:
        ratio = float(target_keep_ratio)
        if not 0.0 < ratio <= 1.0:
            raise ValueError("target_keep_ratio must be in (0, 1]")
        k = max(1, min(n_reactions, int(round(n_reactions * ratio))))
        order = np.argsort(-probs)
        keep_target = np.zeros(n_reactions, dtype=bool)
        keep_target[order[:k]] = True
        if enforce_target_exact:
            keep = keep_target
        else:
            keep = keep | keep_target
        if status == "ok":
            status = "ok_targeted"
        else:
            status = f"{status}_targeted"

    keep = np.asarray(keep, dtype=bool)
    if min_keep_count is not None:
        mk = max(1, min(int(min_keep_count), n_reactions))
        if int(np.sum(keep)) < mk:
            order = np.argsort(-probs)
            keep = np.zeros_like(keep, dtype=bool)
            keep[order[:mk]] = True
            status = "forced_min_keep"

    if not np.any(keep):
        status = "fallback_all_pruned"
        importance = np.sum(np.abs(rop_arr), axis=0)
        keep[int(np.argmax(importance))] = True

    if return_details:
        details: dict[str, Any] = {
            "status": status,
            "keep_count": int(np.sum(keep)),
            "keep_ratio": float(np.mean(keep)),
            "gate_probs": probs.tolist(),
            "lambda_l0": float(lambda_l0),
            "threshold": float(threshold),
            "target_keep_ratio": (None if target_keep_ratio is None else float(target_keep_ratio)),
            "enforce_target_exact": bool(enforce_target_exact),
            "min_keep_count": (None if min_keep_count is None else int(min_keep_count)),
        }
        return keep, details
    return keep
