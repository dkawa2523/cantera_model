from __future__ import annotations

from typing import Any

import numpy as np


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.sign(arr) * np.log1p(np.abs(arr))


def _signed_expm1(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.sign(arr) * np.expm1(np.abs(arr))


def fit_rate_model(features: np.ndarray, rates_target: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    X = np.asarray(features, dtype=float)
    Y = np.asarray(rates_target, dtype=float)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("features and rates_target must be 2-D")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("features rows must match rates_target rows")

    model_name = str((cfg or {}).get("model", "log_linear"))
    if model_name != "log_linear":
        raise ValueError("rate.model must be 'log_linear'")

    l2 = float((cfg or {}).get("l2", 1.0e-6))
    l2 = max(l2, 0.0)

    Xt = _signed_log1p(Y)
    xtx = X.T @ X
    reg = l2 * np.eye(xtx.shape[0], dtype=float)
    coef = np.linalg.pinv(xtx + reg) @ (X.T @ Xt)

    return {
        "model_type": "log_linear",
        "feature_dim": int(X.shape[1]),
        "output_dim": int(Y.shape[1]),
        "coef": coef.tolist(),
        "l2": float(l2),
    }


def predict_rates(model_artifact: dict[str, Any], features: np.ndarray) -> np.ndarray:
    art = dict(model_artifact or {})
    if str(art.get("model_type", "")) != "log_linear":
        raise ValueError("model_artifact.model_type must be 'log_linear'")

    X = np.asarray(features, dtype=float)
    coef = np.asarray(art.get("coef"), dtype=float)
    if X.ndim != 2 or coef.ndim != 2:
        raise ValueError("features and coef must be 2-D")
    if X.shape[1] != coef.shape[0]:
        raise ValueError("features columns must match model feature_dim")

    pred_log = X @ coef
    return _signed_expm1(pred_log)
