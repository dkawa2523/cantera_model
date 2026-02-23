from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np

from cantera_model.eval.cantera_runner import compare_rows
from cantera_model.types import EvalSummary


def _synthetic_metric(case: dict[str, Any], key: str) -> float:
    phi = float(case.get("phi", 1.0))
    t_end = float(case.get("t_end", 0.1))
    T0 = float(case.get("T0", 1000.0))

    if key == "ignition_delay":
        return t_end / max(0.1, 1.0 + phi)
    if key == "T_last":
        return T0 + 20.0 * phi
    if key == "T_max":
        return T0 + 60.0 * phi

    if key.startswith("X_last:"):
        name = key.split(":", 1)[1]
        base = (sum(ord(c) for c in name) % 17 + 1) / 200.0
        return base / max(1.0, phi)
    if key.startswith("X_max:"):
        name = key.split(":", 1)[1]
        base = (sum(ord(c) for c in name) % 13 + 1) / 100.0
        return min(1.0, base * (0.5 + 0.3 * phi))
    if key.startswith("X_int:"):
        name = key.split(":", 1)[1]
        base = (sum(ord(c) for c in name) % 19 + 1) / 150.0
        return max(0.0, base * max(t_end, 1.0e-6))
    if key.startswith("dep_int:"):
        name = key.split(":", 1)[1]
        base = (sum(ord(c) for c in name) % 23 + 1) / 200.0
        return max(0.0, base * max(t_end, 1.0e-6))
    return 0.0


def _default_metric_keys(qoi_cfg: dict[str, Any]) -> list[str]:
    species_last = list(qoi_cfg.get("species_last") or [])
    species_max = list(qoi_cfg.get("species_max") or [])
    species_integral = list(qoi_cfg.get("species_integral") or [])
    deposition_integral = list(qoi_cfg.get("deposition_integral") or [])

    keys = ["ignition_delay", "T_max", "T_last"]
    keys.extend([f"X_last:{sp}" for sp in species_last])
    keys.extend([f"X_max:{sp}" for sp in species_max])
    keys.extend([f"X_int:{sp}" for sp in species_integral])
    keys.extend([f"dep_int:{sp}" for sp in deposition_integral])
    uniq: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        uniq.append(key)
    return uniq


def _case_features(case: dict[str, Any]) -> np.ndarray:
    T0 = float(case.get("T0", 1000.0))
    P0 = float(case.get("P0_atm", 1.0))
    phi = float(case.get("phi", 1.0))
    t_end = float(case.get("t_end", 0.1))
    return np.asarray(
        [
            1.0,
            T0,
            P0,
            phi,
            t_end,
            T0 * phi,
            phi * t_end,
            np.log1p(max(P0, 0.0)),
        ],
        dtype=float,
    )


def fit_lightweight_surrogate(
    conditions: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    qoi_cfg: dict[str, Any],
    *,
    l2: float = 1.0e-6,
    split_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not conditions or not baseline_rows:
        raise ValueError("conditions and baseline_rows must be non-empty")

    by_case = {str(r["case_id"]): r for r in baseline_rows}
    matched = [c for c in conditions if str(c.get("case_id")) in by_case]
    if not matched:
        raise ValueError("no overlapping case_id between conditions and baseline_rows")

    split = dict(split_cfg or {})
    requested_mode = str(split.get("mode", "in_sample"))
    mode = requested_mode
    fallback_reason: str | None = None

    if mode not in {"in_sample", "holdout"}:
        raise ValueError("split_cfg.mode must be 'in_sample' or 'holdout'")

    train_cases = list(matched)
    test_case_ids: list[str] = []
    if mode == "holdout":
        holdout_ratio = float(split.get("holdout_ratio", 0.25))
        holdout_ratio = float(np.clip(holdout_ratio, 0.0, 0.9))
        min_train_cases = int(split.get("min_train_cases", 2))
        ordered = sorted(matched, key=lambda c: str(c.get("case_id")))
        n_total = len(ordered)
        n_test = int(max(1, round(n_total * holdout_ratio)))
        n_test = min(n_test, max(1, n_total - 1))
        if n_total - n_test < max(1, min_train_cases):
            mode = "in_sample"
            fallback_reason = "insufficient_cases_for_holdout"
        else:
            test_cases = ordered[-n_test:]
            test_case_ids = [str(c.get("case_id")) for c in test_cases]
            train_cases = ordered[:-n_test]

    X = np.vstack([_case_features(c) for c in train_cases])
    metric_keys = _default_metric_keys(qoi_cfg)

    xtx = X.T @ X
    reg = float(max(l2, 0.0)) * np.eye(xtx.shape[0], dtype=float)
    inv = np.linalg.pinv(xtx + reg)

    metric_models: dict[str, Any] = {}
    for key in metric_keys:
        y = np.asarray([float(by_case[str(c["case_id"])].get(key, _synthetic_metric(c, key))) for c in train_cases], dtype=float)
        coef = inv @ (X.T @ y)
        metric_models[key] = {
            "coef": coef.tolist(),
            "mean": float(np.mean(y)),
        }

    return {
        "model_type": "linear_ridge",
        "feature_order": ["bias", "T0", "P0_atm", "phi", "t_end", "T0*phi", "phi*t_end", "log1p(P0_atm)"],
        "metric_models": metric_models,
        "l2": float(l2),
        "split_meta": {
            "requested_mode": requested_mode,
            "mode": mode,
            "fallback_reason": fallback_reason,
            "train_case_ids": [str(c.get("case_id")) for c in train_cases],
            "test_case_ids": test_case_ids,
        },
    }


def _predict_linear_surrogate(linear_surrogate: dict[str, Any], case: dict[str, Any], key: str) -> float:
    models = dict(linear_surrogate.get("metric_models") or {})
    model = dict(models.get(key) or {})
    coef = np.asarray(model.get("coef") or [], dtype=float)
    if coef.size == 0:
        return _synthetic_metric(case, key)
    x = _case_features(case)
    if coef.shape != x.shape:
        return _synthetic_metric(case, key)
    return float(np.dot(x, coef))


def _deterministic_signed(case_id: str, key: str, seed: int) -> float:
    raw = f"{case_id}|{key}|{seed}".encode("utf-8")
    digest = hashlib.blake2b(raw, digest_size=8).digest()
    val = int.from_bytes(digest, byteorder="big", signed=False)
    return (val / float(2**64 - 1)) * 2.0 - 1.0


def run_surrogate_cases(
    model_artifact: dict[str, Any] | Callable[[dict[str, Any], dict[str, Any]], dict[str, float]],
    conditions: list[dict[str, Any]],
    qoi_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    if callable(model_artifact):
        out = []
        for case in conditions:
            metrics = model_artifact(case, qoi_cfg)
            out.append({"case_id": case["case_id"], **metrics})
        return out

    keys = _default_metric_keys(qoi_cfg)
    ref_rows = {str(r["case_id"]): r for r in (model_artifact.get("reference_rows") or [])}
    linear_surrogate = model_artifact.get("linear_surrogate")
    blend_reference = float(model_artifact.get("blend_reference", 0.0))
    preserve_small_reference = bool(model_artifact.get("preserve_small_reference", True))
    small_reference_eps = float(model_artifact.get("small_reference_eps", 1.0e-10))
    perturb_scale = float(model_artifact.get("perturb_scale", 0.0))
    perturb_seed = int(model_artifact.get("perturb_seed", 0))
    global_scale = float(model_artifact.get("global_scale", 1.0))
    metric_scale = dict(model_artifact.get("metric_scale") or {})
    metric_bias = dict(model_artifact.get("metric_bias") or {})

    rows: list[dict[str, Any]] = []
    for case in conditions:
        row: dict[str, Any] = {"case_id": case["case_id"]}
        ref = ref_rows.get(str(case["case_id"]))
        for key in keys:
            if linear_surrogate is not None:
                pred_val = _predict_linear_surrogate(dict(linear_surrogate), case, key)
                if ref is not None and key in ref and blend_reference > 0.0:
                    alpha = float(np.clip(blend_reference, 0.0, 1.0))
                    base_val = alpha * float(ref[key]) + (1.0 - alpha) * pred_val
                else:
                    base_val = pred_val
            elif ref is not None and key in ref:
                base_val = float(ref[key])
            else:
                base_val = _synthetic_metric(case, key)
            scale = float(metric_scale.get(key, global_scale))
            bias = float(metric_bias.get(key, 0.0))
            value = base_val * scale + bias
            if preserve_small_reference and ref is not None and key in ref:
                ref_val = float(ref[key])
                if abs(ref_val) <= small_reference_eps:
                    row[key] = ref_val
                    continue
            if perturb_scale > 0.0:
                signed = _deterministic_signed(str(case["case_id"]), key, perturb_seed)
                value = value * (1.0 + perturb_scale * signed)
            if key.startswith("X_") or key.startswith("dep_") or key in {"ignition_delay", "T_max", "T_last"}:
                value = max(0.0, value)
            row[key] = value
        rows.append(row)
    return rows


def run_surrogate_traces(
    model_artifact: dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]],
    traces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if callable(model_artifact):
        return [model_artifact(t) for t in traces]

    out: list[dict[str, Any]] = []
    global_scale = float(model_artifact.get("global_scale", 1.0))
    wdot_scale = float(model_artifact.get("wdot_scale", global_scale))
    wdot_bias = float(model_artifact.get("wdot_bias", 0.0))

    for trace in traces:
        time = np.asarray(trace.get("time"), dtype=float)
        x_ref = np.asarray(trace.get("X"), dtype=float)
        wdot_ref = np.asarray(trace.get("wdot"), dtype=float)
        if x_ref.ndim != 2 or wdot_ref.ndim != 2 or x_ref.shape != wdot_ref.shape:
            raise ValueError("trace rows must include X/wdot with matching 2-D shapes")
        if time.ndim != 1 or time.shape[0] != x_ref.shape[0]:
            raise ValueError("trace rows must include time with matching first dimension")

        dt = np.zeros_like(time)
        if time.size > 1:
            dt[1:] = np.maximum(np.diff(time), 0.0)

        wdot_pred = (wdot_ref * wdot_scale) + wdot_bias
        x_pred = np.zeros_like(x_ref)
        x_pred[0] = np.maximum(x_ref[0], 0.0)
        for i in range(1, x_pred.shape[0]):
            x_pred[i] = np.maximum(0.0, x_pred[i - 1] + wdot_pred[i - 1] * dt[i])

        out.append(
            {
                "case_id": str(trace.get("case_id")),
                "time": time,
                "X": x_pred,
                "wdot": wdot_pred,
            }
        )
    return out


def compare_with_baseline(
    baseline_rows: list[dict[str, Any]],
    surrogate_rows: list[dict[str, Any]],
    eval_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], EvalSummary]:
    rel_eps = float(eval_cfg.get("rel_eps", 1.0e-12))
    rel_tolerance = float(eval_cfg.get("rel_tolerance", 0.2))

    cmp_rows, summary = compare_rows(
        baseline_rows,
        surrogate_rows,
        rel_eps=rel_eps,
        rel_tolerance=rel_tolerance,
    )

    out = EvalSummary(
        cases=int(summary["cases"]),
        pass_rate=float(summary["pass_rate"]),
        pass_cases=int(summary["pass_cases"]),
        failed_cases=int(summary["failed_cases"]),
        qoi_metrics_count=int(summary["qoi_metrics_count"]),
        max_rel_diff=(None if summary["max_rel_diff"] is None else float(summary["max_rel_diff"])),
        mean_rel_diff=(None if summary["mean_rel_diff"] is None else float(summary["mean_rel_diff"])),
        worst_case=summary.get("worst_case"),
        rel_tolerance=rel_tolerance,
        rel_eps=rel_eps,
    )
    return cmp_rows, out
