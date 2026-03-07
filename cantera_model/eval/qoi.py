from __future__ import annotations

import math
from typing import Any

import numpy as np

def _series_stat(values: list[float], stat: str, time: list[float] | None = None) -> float:
    if not values:
        return float("nan")
    if stat == "final":
        return float(values[-1])
    if stat == "mean":
        return float(sum(values) / len(values))
    if stat == "max":
        return float(max(values))
    if stat == "min":
        return float(min(values))
    if stat == "integral":
        arr = np.maximum(np.asarray(values, dtype=float), 0.0)
        if arr.size == 0:
            return 0.0
        if time is None:
            return float(np.trapezoid(arr))
        t = np.asarray(time, dtype=float)
        if t.shape != arr.shape:
            return float(np.trapezoid(arr))
        if arr.size <= 1:
            return float(arr[-1])
        return float(np.trapezoid(arr, t))
    raise ValueError(f"unsupported stat: {stat}")


def _selector_value(selector: str, row_ctx: dict[str, Any]) -> float:
    parts = selector.split(":")
    if len(parts) != 3:
        raise ValueError(f"invalid QoI selector: {selector}")
    domain, name, stat = parts
    stat = stat.strip().lower()

    if domain == "gas_X":
        species_names = list(row_ctx.get("gas_species_names") or [])
        x_rows = list(row_ctx.get("gas_X") or [])
        time = list(row_ctx.get("time") or [])
        try:
            idx = species_names.index(name)
        except ValueError:
            return float("nan")
        series = [float(max(0.0, row[idx])) for row in x_rows]
        return _series_stat(series, stat, time=time)

    if domain == "surface_theta":
        species_names = list(row_ctx.get("surface_species_names") or [])
        theta_rows = list(row_ctx.get("surface_theta") or [])
        time = list(row_ctx.get("time") or [])
        try:
            idx = species_names.index(name)
        except ValueError:
            return float("nan")
        series = [float(max(0.0, row[idx])) for row in theta_rows]
        return _series_stat(series, stat, time=time)

    if domain == "deposition_rate":
        dep = dict(row_ctx.get("deposition_rate") or {})
        time = list(row_ctx.get("time") or [])
        series = [float(v) for v in (dep.get(name) or [])]
        return _series_stat(series, stat, time=time)

    raise ValueError(f"unsupported QoI selector domain: {domain}")


def extract_qoi(row_ctx: dict[str, Any], qoi_cfg: dict[str, Any] | None) -> dict[str, float]:
    cfg = dict(qoi_cfg or {})
    selectors = cfg.get("selectors")
    if selectors is None:
        return {}
    out: dict[str, float] = {}
    for raw in selectors:
        sel = str(raw).strip()
        if not sel:
            continue
        try:
            value = _selector_value(sel, row_ctx)
        except ValueError:
            value = float("nan")
        out[sel] = value if math.isfinite(value) or math.isnan(value) else float("nan")
    return out
