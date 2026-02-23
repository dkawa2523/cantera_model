from __future__ import annotations

import numpy as np


def _as_2d(x: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr[None, :], True
    if arr.ndim != 2:
        raise ValueError("state array must be 1-D or 2-D")
    return arr, False


def project_to_conservation(
    X: np.ndarray,
    A: np.ndarray,
    *,
    reference: np.ndarray | None = None,
    clip_nonnegative: bool = False,
    max_iter: int = 2,
) -> np.ndarray:
    arr, squeeze = _as_2d(X)
    A_mat = np.asarray(A, dtype=float)
    if A_mat.ndim != 2:
        raise ValueError("A must be 2-D")
    if arr.shape[1] != A_mat.shape[1]:
        raise ValueError("X and A species dimension mismatch")

    if reference is None:
        ref = arr[0]
    else:
        ref = np.asarray(reference, dtype=float)
        if ref.shape != (arr.shape[1],):
            raise ValueError("reference must have shape (Ns,)")

    b = A_mat @ ref
    gram = A_mat @ A_mat.T

    out = arr.copy()
    for t in range(out.shape[0]):
        x = out[t]
        for _ in range(max_iter):
            resid = (A_mat @ x) - b
            if float(np.max(np.abs(resid))) < 1.0e-12:
                break
            delta = np.linalg.pinv(gram) @ resid
            x = x - A_mat.T @ delta
            if clip_nonnegative:
                x = np.maximum(x, 0.0)
        out[t] = x

    return out[0] if squeeze else out


def conservation_violation(
    X: np.ndarray,
    A: np.ndarray,
    *,
    reference: np.ndarray | None = None,
) -> float:
    arr, _ = _as_2d(X)
    A_mat = np.asarray(A, dtype=float)
    ref = arr[0] if reference is None else np.asarray(reference, dtype=float)
    target = A_mat @ ref
    residual = (A_mat @ arr.T).T - target[None, :]
    return float(np.max(np.abs(residual)))
