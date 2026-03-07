#!/usr/bin/env python3
"""Standalone Cantera evaluator for gas and surface benchmark conditions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cantera_model.eval.conditions import load_conditions as _load_conditions
from cantera_model.eval.qoi import extract_qoi

try:
    import cantera as ct
except ImportError:  # pragma: no cover - handled at runtime where needed
    ct = None


def _require_cantera() -> None:
    if ct is None:
        raise ImportError("cantera is required for this operation")


def _as_float(value: Any, field: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid float for {field}: {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"non-finite value for {field}: {value!r}")
    return out


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    return data


def load_conditions(
    path: Path,
    mode: str = "gas_homogeneous",
    schema_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return _load_conditions(path, mode=mode, schema_cfg=schema_cfg)


def mechanism_counts(
    mech_ref: str,
    phase: str,
    *,
    mode: str = "gas_homogeneous",
    gas_phase: str = "gas",
    interface_phase: str | None = None,
) -> dict[str, int]:
    _require_cantera()
    if mode == "gas_homogeneous":
        gas = ct.Solution(mech_ref, phase)
        return {"species": int(gas.n_species), "reactions": int(gas.n_reactions)}

    if mode == "surface_batch":
        if not interface_phase:
            raise ValueError("simulation.surface.interface_phase is required for surface mode")
        iface, gas = _load_surface_or_raise(mech_ref, interface_phase=interface_phase, gas_phase=gas_phase)
        return {
            "species": int(gas.n_species + iface.n_species),
            "reactions": int(gas.n_reactions + iface.n_reactions),
            "gas_species": int(gas.n_species),
            "gas_reactions": int(gas.n_reactions),
            "surface_species": int(iface.n_species),
            "surface_reactions": int(iface.n_reactions),
        }

    raise ValueError(f"unsupported simulation mode: {mode}")


def _load_mechanism_or_raise(mech_ref: str, phase: str) -> Any:
    _require_cantera()
    return ct.Solution(mech_ref, phase)


def _load_surface_or_raise(mech_ref: str, *, interface_phase: str, gas_phase: str) -> tuple[Any, Any]:
    _require_cantera()
    iface = ct.Interface(mech_ref, interface_phase)

    gas = None
    adjacent = iface.adjacent
    if gas_phase in adjacent:
        gas = adjacent[gas_phase]
    else:
        for key, phase_obj in adjacent.items():
            if key.lower() == gas_phase.lower() or getattr(phase_obj, "name", "").lower() == gas_phase.lower():
                gas = phase_obj
                break
    if gas is None:
        raise ValueError(f"gas phase not found in interface.adjacent: {gas_phase}")
    return iface, gas


def _phase_name_from_config(cfg: dict[str, Any]) -> str:
    baseline_cfg = cfg.get("baseline") or {}
    return str(baseline_cfg.get("phase", "gri30"))


def _resolve_input_path(raw: Any, *, config_parent: Path) -> Path:
    if raw is None:
        raise ValueError("path value cannot be null")
    path = Path(str(raw))
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (config_parent / path).resolve()


def _resolve_mechanism_ref(raw: Any, *, config_parent: Path) -> str:
    if raw is None:
        raise ValueError("mechanism value cannot be null")
    text = str(raw).strip()
    if not text:
        raise ValueError("mechanism value cannot be empty")
    path = Path(text)
    if path.is_absolute():
        return str(path.resolve())

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)

    cfg_candidate = (config_parent / path).resolve()
    if cfg_candidate.exists():
        return str(cfg_candidate)

    # Allow built-in Cantera data mechanism names (e.g., diamond.yaml).
    if len(path.parts) == 1:
        return text

    return str(cfg_candidate)


def _surface_schema_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    simulation = cfg.get("simulation") or {}
    surface = simulation.get("surface") or {}
    schema = dict(simulation.get("conditions_schema") or {})
    for key in (
        "pressure_unit",
        "pressure_column",
        "temperature_column",
        "composition_column",
        "time_column",
        "n_steps_column",
        "area_column",
    ):
        if key in surface and key not in schema:
            schema[key] = surface[key]
    return schema


def _surface_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    simulation = cfg.get("simulation") or {}
    surface = dict(simulation.get("surface") or {})
    if "interface_phase" not in surface:
        baseline = cfg.get("baseline") or {}
        if "interface_phase" in baseline:
            surface["interface_phase"] = baseline["interface_phase"]
    surface.setdefault("gas_phase", "gas")
    return surface


def _surface_qoi_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    eval_cfg = cfg.get("evaluation") or {}
    qoi_cfg = eval_cfg.get("qoi")
    if isinstance(qoi_cfg, dict):
        return qoi_cfg
    qoi_top = cfg.get("qoi")
    if isinstance(qoi_top, dict):
        return qoi_top
    return {}


def _preflight(
    *,
    baseline_mech: str,
    candidate_mech: str | None,
    phase: str,
    conditions_csv: Path,
    mode: str,
    surface_cfg: dict[str, Any],
    schema_cfg: dict[str, Any],
) -> None:
    _ = load_conditions(conditions_csv, mode=mode, schema_cfg=schema_cfg)

    if mode == "gas_homogeneous":
        _ = _load_mechanism_or_raise(baseline_mech, phase)
        if candidate_mech is not None:
            _ = _load_mechanism_or_raise(candidate_mech, phase)
        return

    if mode == "surface_batch":
        interface_phase = str(surface_cfg.get("interface_phase") or "")
        gas_phase = str(surface_cfg.get("gas_phase", "gas"))
        if not interface_phase:
            raise ValueError("simulation.surface.interface_phase is required")
        _ = _load_surface_or_raise(baseline_mech, interface_phase=interface_phase, gas_phase=gas_phase)
        if candidate_mech is not None:
            _ = _load_surface_or_raise(candidate_mech, interface_phase=interface_phase, gas_phase=gas_phase)
        return

    raise ValueError(f"unsupported simulation mode: {mode}")


def _ignition_delay(times: list[float], temps: list[float]) -> float:
    if len(times) < 3:
        return float("nan")
    max_slope = -float("inf")
    max_time = float("nan")
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if dt <= 0.0:
            continue
        slope = (temps[i] - temps[i - 1]) / dt
        if slope > max_slope:
            max_slope = slope
            max_time = times[i]
    return max_time


def _simulate_case_series(
    mech_ref: str,
    phase: str,
    case: dict[str, Any],
    *,
    fuel: str,
    oxidizer: str,
    n_steps: int,
) -> dict[str, Any]:
    _require_cantera()
    gas = ct.Solution(mech_ref, phase)
    gas.set_equivalence_ratio(case["phi"], fuel, oxidizer)
    gas.TP = case["T0"], case["P0_atm"] * ct.one_atm

    reactor = ct.IdealGasConstPressureReactor(gas)
    net = ct.ReactorNet([reactor])
    t_end = max(case["t_end"], 1.0e-12)
    n_steps = max(int(n_steps), 2)

    times = [0.0]
    temps = [float(reactor.T)]
    press = [float(reactor.thermo.P)]
    x_rows = [reactor.thermo.X.copy()]
    wdot_rows = [reactor.thermo.net_production_rates.copy()]
    rop_rows = [reactor.kinetics.net_rates_of_progress.copy()]

    for i in range(n_steps):
        target = t_end * float(i + 1) / float(n_steps)
        net.advance(target)
        times.append(float(net.time))
        temps.append(float(reactor.T))
        press.append(float(reactor.thermo.P))
        x_rows.append(reactor.thermo.X.copy())
        wdot_rows.append(reactor.thermo.net_production_rates.copy())
        rop_rows.append(reactor.kinetics.net_rates_of_progress.copy())

    return {
        "species_names": list(reactor.thermo.species_names),
        "reaction_eqs": list(reactor.kinetics.reaction_equations()),
        "time": times,
        "temperature": temps,
        "pressure": press,
        "X": x_rows,
        "wdot": wdot_rows,
        "rop": rop_rows,
    }


def _deposition_labels_from_qoi(qoi_cfg: dict[str, Any]) -> list[str]:
    selectors = list(qoi_cfg.get("selectors") or [])
    labels: list[str] = []
    for selector in selectors:
        parts = str(selector).split(":")
        if len(parts) != 3:
            continue
        if parts[0] == "deposition_rate":
            labels.append(parts[1])
    seen: set[str] = set()
    uniq: list[str] = []
    for item in labels:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq


def _simulate_case_surface_series(
    mech_ref: str,
    case: dict[str, Any],
    *,
    gas_phase: str,
    interface_phase: str,
    default_n_steps: int,
    default_area: float,
    qoi_cfg: dict[str, Any],
) -> dict[str, Any]:
    _require_cantera()
    iface, gas = _load_surface_or_raise(mech_ref, interface_phase=interface_phase, gas_phase=gas_phase)

    n_steps = int(case.get("n_steps", default_n_steps))
    n_steps = max(n_steps, 2)
    area = float(case.get("area", default_area))
    area = max(area, 1.0e-12)

    gas.TPX = case["T_K"], case["P_Pa"], case["composition"]

    reactor = ct.IdealGasConstPressureReactor(gas, energy="off")
    _ = ct.ReactorSurface(iface, reactor, A=area)
    net = ct.ReactorNet([reactor])
    t_end = max(float(case["t_end_s"]), 1.0e-12)

    dep_labels = _deposition_labels_from_qoi(qoi_cfg)
    kinetics_species = list(iface.kinetics_species_names)
    dep_index = {label: kinetics_species.index(label) for label in dep_labels if label in kinetics_species}

    times = [0.0]
    gas_x = [gas.X.copy()]
    theta = [iface.coverages.copy()]
    dep_series: dict[str, list[float]] = {label: [] for label in dep_labels}

    rates = iface.net_production_rates
    for label in dep_labels:
        idx = dep_index.get(label)
        dep_series[label].append(float(rates[idx]) if idx is not None else float("nan"))

    for i in range(n_steps):
        target = t_end * float(i + 1) / float(n_steps)
        net.advance(target)
        times.append(float(net.time))
        gas_x.append(gas.X.copy())
        theta.append(iface.coverages.copy())

        rates = iface.net_production_rates
        for label in dep_labels:
            idx = dep_index.get(label)
            dep_series[label].append(float(rates[idx]) if idx is not None else float("nan"))

    return {
        "time": times,
        "gas_species_names": list(gas.species_names),
        "surface_species_names": list(iface.species_names),
        "gas_X": gas_x,
        "surface_theta": theta,
        "deposition_rate": dep_series,
    }


def _compute_surface_qois(series: dict[str, Any], qoi_cfg: dict[str, Any]) -> dict[str, float]:
    row_ctx = {
        "time": list(series.get("time") or []),
        "gas_species_names": list(series.get("gas_species_names") or []),
        "surface_species_names": list(series.get("surface_species_names") or []),
        "gas_X": list(series.get("gas_X") or []),
        "surface_theta": list(series.get("surface_theta") or []),
        "deposition_rate": dict(series.get("deposition_rate") or {}),
    }
    return extract_qoi(row_ctx, qoi_cfg)


def run_case(
    mech_ref: str,
    phase: str,
    case: dict[str, Any],
    *,
    fuel: str,
    oxidizer: str,
    n_steps: int,
    species_last: list[str],
    species_max: list[str],
    include_temperature_metrics: bool = True,
    include_ignition_delay: bool = True,
) -> dict[str, float]:
    series = _simulate_case_series(
        mech_ref,
        phase,
        case,
        fuel=fuel,
        oxidizer=oxidizer,
        n_steps=n_steps,
    )
    times = list(series["time"])
    temps = list(series["temperature"])
    x_mat = series["X"]
    species_names = list(series["species_names"])

    max_temp = max(float(t) for t in temps)
    last_x = x_mat[-1]
    max_species: dict[str, float] = {}
    name_to_idx = {n: i for i, n in enumerate(species_names)}
    for sp in species_max:
        idx = name_to_idx.get(sp)
        if idx is None:
            max_species[sp] = float("nan")
        else:
            max_species[sp] = max(float(max(0.0, row[idx])) for row in x_mat)

    out: dict[str, float] = {}
    if include_ignition_delay:
        out["ignition_delay"] = _ignition_delay(times, temps)
    if include_temperature_metrics:
        out["T_max"] = max_temp
        out["T_last"] = float(temps[-1])

    for sp in species_last:
        key = f"X_last:{sp}"
        idx = name_to_idx.get(sp)
        out[key] = float(max(0.0, last_x[idx])) if idx is not None else float("nan")
    for sp in species_max:
        key = f"X_max:{sp}"
        out[key] = float(max_species.get(sp, float("nan")))
    return out


def run_case_trace(
    mech_ref: str,
    phase: str,
    case: dict[str, Any],
    *,
    fuel: str,
    oxidizer: str,
    n_steps: int,
    mode: str = "gas_homogeneous",
    surface_cfg: dict[str, Any] | None = None,
    qoi_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if mode == "gas_homogeneous":
        return _simulate_case_series(
            mech_ref,
            phase,
            case,
            fuel=fuel,
            oxidizer=oxidizer,
            n_steps=n_steps,
        )

    if mode == "surface_batch":
        s_cfg = dict(surface_cfg or {})
        return _simulate_case_surface_series(
            mech_ref,
            case,
            gas_phase=str(s_cfg.get("gas_phase", "gas")),
            interface_phase=str(s_cfg.get("interface_phase") or ""),
            default_n_steps=n_steps,
            default_area=float(s_cfg.get("default_area", 1.0)),
            qoi_cfg=dict(qoi_cfg or {}),
        )

    raise ValueError(f"unsupported simulation mode: {mode}")


def run_case_surface(
    mech_ref: str,
    case: dict[str, Any],
    *,
    surface_cfg: dict[str, Any],
    n_steps: int,
    qoi_cfg: dict[str, Any],
) -> dict[str, float]:
    series = _simulate_case_surface_series(
        mech_ref,
        case,
        gas_phase=str(surface_cfg.get("gas_phase", "gas")),
        interface_phase=str(surface_cfg.get("interface_phase") or ""),
        default_n_steps=n_steps,
        default_area=float(surface_cfg.get("default_area", 1.0)),
        qoi_cfg=qoi_cfg,
    )
    return _compute_surface_qois(series, qoi_cfg)


def run_mechanism(
    mech_ref: str,
    phase: str,
    conditions: list[dict[str, Any]],
    *,
    fuel: str,
    oxidizer: str,
    n_steps: int,
    species_last: list[str],
    species_max: list[str],
    mode: str = "gas_homogeneous",
    surface_cfg: dict[str, Any] | None = None,
    qoi_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    qoi_runtime_cfg = dict(qoi_cfg or {})
    builtins_cfg = dict(qoi_runtime_cfg.get("qoi_builtin_metrics") or {})
    include_temperature_metrics = bool(builtins_cfg.get("include_temperature_metrics", True))
    include_ignition_delay = bool(builtins_cfg.get("include_ignition_delay", True))
    if mode == "gas_homogeneous":
        for case in conditions:
            metrics = run_case(
                mech_ref,
                phase,
                case,
                fuel=fuel,
                oxidizer=oxidizer,
                n_steps=n_steps,
                species_last=species_last,
                species_max=species_max,
                include_temperature_metrics=include_temperature_metrics,
                include_ignition_delay=include_ignition_delay,
            )
            rows.append({"case_id": case["case_id"], **metrics})
        return rows

    if mode == "surface_batch":
        surface = dict(surface_cfg or {})
        qoi = dict(qoi_cfg or {})
        for case in conditions:
            metrics = run_case_surface(
                mech_ref,
                case,
                surface_cfg=surface,
                n_steps=n_steps,
                qoi_cfg=qoi,
            )
            rows.append({"case_id": case["case_id"], **metrics})
        return rows

    raise ValueError(f"unsupported simulation mode: {mode}")


def _metric_family(metric_key: str, metric_taxonomy: dict[str, Any] | None = None) -> str:
    key = str(metric_key)
    taxonomy = dict(metric_taxonomy or {})
    family_exact = dict(taxonomy.get("family_exact") or {})
    if key in family_exact:
        return str(family_exact.get(key) or "default")
    family_prefix = dict(taxonomy.get("family_prefix") or {})
    for prefix, family in family_prefix.items():
        if prefix and key.startswith(str(prefix)):
            return str(family or "default")
    if key.startswith("X_last:"):
        return "X_last"
    if key.startswith("X_max:"):
        return "X_max"
    if key.startswith("X_int:"):
        return "X_int"
    if key.startswith("dep_int:"):
        return "dep_int"
    if key in {"T_max", "T_last"}:
        return "T"
    if key == "ignition_delay":
        return "ignition_delay"
    return "default"


def _metric_species_token(metric_key: str, metric_taxonomy: dict[str, Any] | None = None) -> str:
    key = str(metric_key)
    taxonomy = dict(metric_taxonomy or {})
    species_cfg = dict(taxonomy.get("species_token") or {})
    delimiter = str(species_cfg.get("delimiter", ":"))
    if delimiter and delimiter in key:
        if str(species_cfg.get("take", "after_first")).strip().lower() == "after_first":
            _, token = key.split(delimiter, 1)
        else:
            token = key.rsplit(delimiter, 1)[-1]
        token = token.strip()
        return token or key
    if ":" not in key:
        return key
    _, token = key.split(":", 1)
    token = token.strip()
    return token or key


def _metric_family_weight(family: str, family_weights: dict[str, Any]) -> float:
    raw_weight = family_weights.get(family, family_weights.get("default", 1.0))
    try:
        weight = float(raw_weight)
    except (TypeError, ValueError):
        weight = 1.0
    if not math.isfinite(weight) or weight <= 0.0:
        return 1.0
    return float(weight)


def _metric_abs_floor(
    metric_key: str,
    metric_normalization: dict[str, Any],
    rel_eps: float,
    metric_taxonomy: dict[str, Any] | None = None,
) -> float:
    taxonomy = dict(metric_taxonomy or {})
    family_floors = dict(
        taxonomy.get("metric_family_abs_floor")
        or metric_normalization.get("metric_family_abs_floor")
        or {}
    )
    family = _metric_family(metric_key, taxonomy)
    raw = family_floors.get(family, family_floors.get("default", 0.0))
    try:
        floor = float(raw or 0.0)
    except (TypeError, ValueError):
        floor = 0.0
    return float(max(rel_eps, floor))


def _safe_rel_diff(a: float, b: float, eps: float, *, abs_floor: float = 0.0) -> float:
    if math.isnan(a) and math.isnan(b):
        return 0.0
    if math.isnan(a) or math.isnan(b):
        return float("inf")
    den = max(abs(a), eps, abs_floor)
    return abs(b - a) / den


def _build_metric_tables(
    *,
    by_case_base: dict[str, dict[str, Any]],
    by_case_cand: dict[str, dict[str, Any]],
    case_ids: list[str],
    metric_keys: list[str],
    rel_eps: float,
    rel_tolerance: float,
    metric_normalization: dict[str, Any],
    metric_taxonomy: dict[str, Any],
    mandatory_metric_key_set: set[str],
    mandatory_fail_cases: dict[str, int],
    mandatory_case_pass_metric_counts: dict[str, int],
) -> dict[str, Any]:
    comparison: list[dict[str, Any]] = []
    per_case_pass: list[bool] = []
    rel_values: list[float] = []
    per_case_metric_pass: dict[str, dict[str, bool]] = {case_id: {} for case_id in case_ids}
    per_case_metric_rel: dict[str, dict[str, float]] = {case_id: {} for case_id in case_ids}
    metric_rel_values: dict[str, list[float]] = {}
    metric_pass_counts: dict[str, int] = {}
    metric_case_counts: dict[str, int] = {}
    metric_low_signal_counts: dict[str, int] = {}

    worst_row: dict[str, Any] | None = None
    worst_rel = -float("inf")
    for case_id in case_ids:
        b = by_case_base[case_id]
        c = by_case_cand[case_id]
        case_ok = True
        for key in metric_keys:
            b_val = float(b.get(key, float("nan")))
            c_val = float(c.get(key, float("nan")))
            abs_floor = _metric_abs_floor(
                key,
                metric_normalization,
                rel_eps,
                metric_taxonomy=metric_taxonomy,
            )
            rel = _safe_rel_diff(
                b_val,
                c_val,
                rel_eps,
                abs_floor=abs_floor,
            )
            passed = bool(math.isfinite(rel) and rel <= rel_tolerance)
            per_case_metric_pass[case_id][key] = passed
            per_case_metric_rel[case_id][key] = float(rel)
            metric_case_counts[key] = int(metric_case_counts.get(key, 0) + 1)
            if passed:
                metric_pass_counts[key] = int(metric_pass_counts.get(key, 0) + 1)
            if abs(b_val) <= abs_floor:
                metric_low_signal_counts[key] = int(metric_low_signal_counts.get(key, 0) + 1)
            if not passed:
                case_ok = False
                if key in mandatory_fail_cases:
                    mandatory_fail_cases[key] += 1
            elif key in mandatory_metric_key_set:
                mandatory_case_pass_metric_counts[case_id] = int(
                    mandatory_case_pass_metric_counts.get(case_id, 0) + 1
                )
            if math.isfinite(rel):
                rel_values.append(rel)
                metric_rel_values.setdefault(key, []).append(rel)
                if rel > worst_rel:
                    worst_rel = rel
                    worst_row = {
                        "case_id": case_id,
                        "metric": key,
                        "rel_diff": rel,
                    }
            comparison.append(
                {
                    "case_id": case_id,
                    "metric": key,
                    "baseline": b_val,
                    "candidate": c_val,
                    "rel_diff": rel,
                    "passed": passed,
                    "denominator_floor": abs_floor,
                }
            )
        per_case_pass.append(case_ok)
    return {
        "comparison": comparison,
        "per_case_pass": per_case_pass,
        "rel_values": rel_values,
        "per_case_metric_pass": per_case_metric_pass,
        "per_case_metric_rel": per_case_metric_rel,
        "metric_rel_values": metric_rel_values,
        "metric_pass_counts": metric_pass_counts,
        "metric_case_counts": metric_case_counts,
        "metric_low_signal_counts": metric_low_signal_counts,
        "worst_row": worst_row,
    }


def _compose_error_gate(
    *,
    error_aggregation_mode: str,
    mandatory_error_passed: bool,
    optional_error_passed: bool,
    mandatory_quality_scope_empty: bool,
    pass_rate_mandatory_case: float,
    mandatory_case_pass_min: float,
    mean_rel_diff_mandatory: float | None,
    max_mean_rel_diff_mandatory: float,
    mandatory_rel_outlier_ratio: float,
    mandatory_outlier_ratio_max_effective: float,
    mandatory_tail_guard_passed: bool,
) -> tuple[bool, str]:
    if error_aggregation_mode != "tiered":
        raise ValueError("error_aggregation.mode must be tiered")
    error_gate_passed = bool(mandatory_error_passed and optional_error_passed)
    error_fail_reason_primary = "none"
    if not error_gate_passed:
        if not mandatory_error_passed:
            if (
                not mandatory_quality_scope_empty
                and pass_rate_mandatory_case + 1.0e-12 < mandatory_case_pass_min
            ):
                error_fail_reason_primary = "mandatory_case_rate"
            elif mean_rel_diff_mandatory is not None and float(mean_rel_diff_mandatory) > max_mean_rel_diff_mandatory:
                error_fail_reason_primary = "mandatory_mean"
            elif float(mandatory_rel_outlier_ratio) > mandatory_outlier_ratio_max_effective:
                error_fail_reason_primary = "mandatory_outlier_ratio"
            elif not bool(mandatory_tail_guard_passed):
                error_fail_reason_primary = "mandatory_tail"
            else:
                error_fail_reason_primary = "mandatory_quality"
        elif not optional_error_passed:
            error_fail_reason_primary = "optional_quality"
    return error_gate_passed, error_fail_reason_primary


def compare_rows(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    rel_eps: float,
    rel_tolerance: float,
    mandatory_validity: dict[str, Any] | None = None,
    eval_policy: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_case_base = {str(r["case_id"]): r for r in baseline_rows}
    by_case_cand = {str(r["case_id"]): r for r in candidate_rows}
    case_ids = sorted(set(by_case_base) & set(by_case_cand))
    if not case_ids:
        raise ValueError("no overlapping cases between baseline and candidate")

    metric_keys = sorted(k for k in baseline_rows[0].keys() if k != "case_id")
    mandatory_cfg = dict(mandatory_validity or {})
    policy_cfg = dict(eval_policy or {})
    error_aggregation = dict(policy_cfg.get("error_aggregation") or {})
    metric_normalization = dict(policy_cfg.get("metric_normalization") or {})
    metric_taxonomy = dict(policy_cfg.get("metric_taxonomy") or {})
    error_aggregation_mode = str(error_aggregation.get("mode", "tiered")).strip().lower() or "tiered"
    if error_aggregation_mode != "tiered":
        raise ValueError(
            "evaluation.error_aggregation.mode must be 'tiered' (legacy modes are removed)"
        )

    denominator_mode = str(metric_normalization.get("denominator_mode", "max_abs_or_floor")).strip().lower()
    if denominator_mode != "max_abs_or_floor":
        raise ValueError(
            "evaluation.metric_normalization.denominator_mode must be 'max_abs_or_floor'"
        )
    low_signal_case_ratio_threshold = float(metric_normalization.get("low_signal_case_ratio_threshold", 1.1) or 1.1)
    low_signal_case_ratio_threshold = float(np.clip(low_signal_case_ratio_threshold, 0.0, 1.0))
    low_signal_policy = str(metric_normalization.get("low_signal_policy", "score_only")).strip().lower() or "score_only"
    if low_signal_policy not in {"score_only", "suppress"}:
        low_signal_policy = "score_only"

    mandatory_names = {str(x) for x in list(mandatory_cfg.get("mandatory_metrics") or []) if str(x)}
    mandatory_prefixes = [str(x) for x in list(mandatory_cfg.get("mandatory_prefixes") or []) if str(x)]
    mandatory_gate_metric_keys = [
        key
        for key in metric_keys
        if key in mandatory_names or any(key.startswith(prefix) for prefix in mandatory_prefixes)
    ]
    if not mandatory_gate_metric_keys and mandatory_names:
        mandatory_gate_metric_keys = [key for key in sorted(mandatory_names) if key in metric_keys]
    mandatory_metric_key_set = set(mandatory_gate_metric_keys)
    optional_names = {str(x) for x in list(policy_cfg.get("optional_metrics") or []) if str(x)}
    optional_prefixes = [str(x) for x in list(policy_cfg.get("optional_prefixes") or []) if str(x)]
    optional_gate_metric_keys = [
        key
        for key in metric_keys
        if key in optional_names or any(key.startswith(prefix) for prefix in optional_prefixes)
    ]
    if not optional_gate_metric_keys:
        optional_gate_metric_keys = [key for key in metric_keys if key not in mandatory_metric_key_set]
    else:
        optional_gate_metric_keys = [key for key in optional_gate_metric_keys if key not in mandatory_metric_key_set]
    optional_metric_key_set = set(optional_gate_metric_keys)
    mandatory_total_metric_count = int(len(mandatory_gate_metric_keys))
    mandatory_fail_cases: dict[str, int] = {key: 0 for key in mandatory_gate_metric_keys}
    mandatory_case_pass_metric_counts: dict[str, int] = {case_id: 0 for case_id in case_ids}
    metric_tables = _build_metric_tables(
        by_case_base=by_case_base,
        by_case_cand=by_case_cand,
        case_ids=case_ids,
        metric_keys=metric_keys,
        rel_eps=rel_eps,
        rel_tolerance=rel_tolerance,
        metric_normalization=metric_normalization,
        metric_taxonomy=metric_taxonomy,
        mandatory_metric_key_set=mandatory_metric_key_set,
        mandatory_fail_cases=mandatory_fail_cases,
        mandatory_case_pass_metric_counts=mandatory_case_pass_metric_counts,
    )
    comparison = list(metric_tables["comparison"])
    per_case_pass = list(metric_tables["per_case_pass"])
    rel_values = list(metric_tables["rel_values"])
    per_case_metric_pass = dict(metric_tables["per_case_metric_pass"])
    per_case_metric_rel = dict(metric_tables["per_case_metric_rel"])
    metric_rel_values = dict(metric_tables["metric_rel_values"])
    metric_pass_counts = dict(metric_tables["metric_pass_counts"])
    metric_case_counts = dict(metric_tables["metric_case_counts"])
    metric_low_signal_counts = dict(metric_tables["metric_low_signal_counts"])
    worst_row = metric_tables["worst_row"]

    pass_cases = int(sum(1 for ok in per_case_pass if ok))
    failed_cases = int(len(per_case_pass) - pass_cases)
    pass_rate = float(pass_cases) / float(len(per_case_pass))
    mandatory_case_pass_min = float(error_aggregation.get("mandatory_case_pass_min", 0.75) or 0.75)

    mandatory_metric_validity_mode_effective = str(
        mandatory_cfg.get("mandatory_metric_validity_mode", "case_pass_rate")
    ).strip().lower()
    if mandatory_metric_validity_mode_effective not in {"case_pass_rate", "all_cases"}:
        mandatory_metric_validity_mode_effective = "case_pass_rate"
    mandatory_metric_case_pass_min_raw = mandatory_cfg.get("mandatory_metric_case_pass_min", None)
    if mandatory_metric_case_pass_min_raw is None:
        mandatory_metric_valid_case_pass_min_effective = float(mandatory_case_pass_min)
    else:
        try:
            mandatory_metric_valid_case_pass_min_effective = float(mandatory_metric_case_pass_min_raw)
        except (TypeError, ValueError):
            mandatory_metric_valid_case_pass_min_effective = float(mandatory_case_pass_min)
    mandatory_metric_valid_case_pass_min_effective = float(
        np.clip(mandatory_metric_valid_case_pass_min_effective, 0.0, 1.0)
    )
    mandatory_validity_basis_effective = str(
        mandatory_cfg.get("mandatory_validity_basis", "coverage_evaluable")
    ).strip().lower()
    if mandatory_validity_basis_effective != "coverage_evaluable":
        raise ValueError(
            "evaluation.gate_metric_validity.mandatory_validity_basis must be 'coverage_evaluable'"
        )
    mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective = float(
        np.clip(
            float(mandatory_cfg.get("mandatory_gate_unit_min_evaluable_case_ratio_shadow", 0.25) or 0.25),
            0.0,
            1.0,
        )
    )
    mandatory_metric_case_pass_rates: dict[str, float] = {}
    for key in mandatory_gate_metric_keys:
        case_count = int(metric_case_counts.get(key, 0))
        pass_count = int(metric_pass_counts.get(key, 0))
        mandatory_metric_case_pass_rates[key] = float(pass_count) / float(max(case_count, 1))
    if mandatory_metric_validity_mode_effective == "all_cases":
        active_invalid_mandatory_metric_keys = [
            key for key, case_pass_rate in mandatory_metric_case_pass_rates.items() if float(case_pass_rate) < (1.0 - 1.0e-12)
        ]
    else:
        active_invalid_mandatory_metric_keys = [
            key
            for key, case_pass_rate in mandatory_metric_case_pass_rates.items()
            if float(case_pass_rate) + 1.0e-12 < mandatory_metric_valid_case_pass_min_effective
        ]
    active_invalid_mandatory_metric_count = int(len(active_invalid_mandatory_metric_keys))
    inactive_mandatory_metric_count = int(max(len(mandatory_names) - mandatory_total_metric_count, 0))
    valid_mandatory_metric_count = int(max(mandatory_total_metric_count - active_invalid_mandatory_metric_count, 0))
    invalid_mandatory_metric_count = int(inactive_mandatory_metric_count + active_invalid_mandatory_metric_count)

    mandatory_valid_unit_mode_effective = str(
        mandatory_cfg.get("mandatory_valid_unit_mode", "species_family_quorum")
    ).strip().lower()
    if mandatory_valid_unit_mode_effective not in {"species_family_quorum", "metric"}:
        mandatory_valid_unit_mode_effective = "species_family_quorum"
    mandatory_species_family_score_mode_effective = str(
        mandatory_cfg.get("mandatory_species_family_score_mode", "uniform")
    ).strip().lower()
    if mandatory_species_family_score_mode_effective not in {"uniform", "weighted"}:
        mandatory_species_family_score_mode_effective = "uniform"
    mandatory_species_family_case_pass_min_effective = float(
        np.clip(float(mandatory_cfg.get("mandatory_species_family_case_pass_min", 0.67) or 0.67), 0.0, 1.0)
    )
    mandatory_family_weights_cfg_raw = error_aggregation.get("mandatory_family_weights")
    mandatory_family_weights_cfg = (
        dict(mandatory_family_weights_cfg_raw) if isinstance(mandatory_family_weights_cfg_raw, dict) else {}
    )
    mandatory_case_unit_weight_mode_effective = str(
        error_aggregation.get("mandatory_case_unit_weight_mode", "uniform")
    ).strip().lower()
    if mandatory_case_unit_weight_mode_effective not in {"uniform", "family_weighted"}:
        mandatory_case_unit_weight_mode_effective = "uniform"
    mandatory_quality_scope_effective = str(
        error_aggregation.get("mandatory_quality_scope", "valid_only")
    ).strip().lower()
    if mandatory_quality_scope_effective not in {"valid_only", "all_units", "hybrid_score"}:
        mandatory_quality_scope_effective = "valid_only"
    mandatory_quality_hard_scope = (
        "all_units" if mandatory_quality_scope_effective == "all_units" else "valid_only"
    )
    mandatory_tail_scope_effective = str(
        error_aggregation.get("mandatory_tail_scope", "quality_scope")
    ).strip().lower()
    if mandatory_tail_scope_effective not in {"quality_scope", "all_units"}:
        mandatory_tail_scope_effective = "quality_scope"
    mandatory_gate_unit_case_pass_rates: dict[str, float] = {}
    mandatory_case_pass_gate_unit_counts: dict[str, int] = {case_id: 0 for case_id in case_ids}
    mandatory_gate_unit_weights: dict[str, float] = {}
    mandatory_case_pass_gate_unit_weights: dict[str, float] = {case_id: 0.0 for case_id in case_ids}
    mandatory_gate_unit_to_metric_keys: dict[str, list[str]] = {}
    mandatory_gate_unit_case_pass: dict[str, dict[str, bool]] = {}

    if mandatory_valid_unit_mode_effective == "metric":
        mandatory_total_gate_unit_count = int(mandatory_total_metric_count)
        mandatory_gate_unit_case_pass_rates = {str(k): float(v) for k, v in mandatory_metric_case_pass_rates.items()}
        for key in mandatory_gate_metric_keys:
            unit_key = str(key)
            mandatory_gate_unit_weights[unit_key] = _metric_family_weight(
                _metric_family(key, metric_taxonomy), mandatory_family_weights_cfg
            )
            mandatory_gate_unit_to_metric_keys[unit_key] = [key]
            mandatory_gate_unit_case_pass[unit_key] = {}
        for case_id in case_ids:
            passed_count = 0
            passed_weight = 0.0
            for key in mandatory_gate_metric_keys:
                unit_key = str(key)
                unit_case_passed = bool(per_case_metric_pass.get(case_id, {}).get(key, False))
                mandatory_gate_unit_case_pass[unit_key][case_id] = bool(unit_case_passed)
                if unit_case_passed:
                    passed_count += 1
                    passed_weight += float(mandatory_gate_unit_weights.get(unit_key, 1.0))
            mandatory_case_pass_gate_unit_counts[case_id] = int(passed_count)
            mandatory_case_pass_gate_unit_weights[case_id] = float(passed_weight)
    else:
        species_family_keys: dict[str, dict[str, list[str]]] = {}
        for key in mandatory_gate_metric_keys:
            species = _metric_species_token(key, metric_taxonomy)
            family = _metric_family(key, metric_taxonomy)
            species_family_keys.setdefault(species, {}).setdefault(family, []).append(key)
        mandatory_total_gate_unit_count = int(len(species_family_keys))
        for species in sorted(species_family_keys.keys()):
            family_map = species_family_keys.get(species, {})
            family_count = int(len(family_map))
            species_metric_keys: list[str] = []
            for family_keys in family_map.values():
                species_metric_keys.extend(list(family_keys))
            mandatory_gate_unit_to_metric_keys[species] = list(dict.fromkeys(species_metric_keys))
            mandatory_gate_unit_case_pass[species] = {}
            mandatory_gate_unit_weights[species] = max(
                (_metric_family_weight(family, mandatory_family_weights_cfg) for family in family_map.keys()),
                default=1.0,
            )
            passed_cases = 0
            for case_id in case_ids:
                if mandatory_species_family_score_mode_effective == "weighted":
                    passed_family_weight = 0.0
                    family_weight_total = 0.0
                    for family, family_keys in family_map.items():
                        family_weight = _metric_family_weight(family, mandatory_family_weights_cfg)
                        family_weight_total += family_weight
                        if any(bool(per_case_metric_pass.get(case_id, {}).get(key, False)) for key in family_keys):
                            passed_family_weight += family_weight
                    family_ratio = float(passed_family_weight) / float(max(family_weight_total, 1.0e-12))
                else:
                    passed_families = 0
                    for family_keys in family_map.values():
                        if any(bool(per_case_metric_pass.get(case_id, {}).get(key, False)) for key in family_keys):
                            passed_families += 1
                    family_ratio = float(passed_families) / float(max(family_count, 1))
                unit_case_passed = bool(family_ratio + 1.0e-12 >= mandatory_species_family_case_pass_min_effective)
                mandatory_gate_unit_case_pass[species][case_id] = bool(unit_case_passed)
                if unit_case_passed:
                    passed_cases += 1
                    mandatory_case_pass_gate_unit_counts[case_id] = int(
                        mandatory_case_pass_gate_unit_counts.get(case_id, 0) + 1
                    )
                    mandatory_case_pass_gate_unit_weights[case_id] = float(
                        mandatory_case_pass_gate_unit_weights.get(case_id, 0.0)
                        + float(mandatory_gate_unit_weights.get(species, 1.0))
                    )
            mandatory_gate_unit_case_pass_rates[species] = float(passed_cases) / float(max(len(case_ids), 1))

    if mandatory_metric_validity_mode_effective == "all_cases":
        active_invalid_mandatory_gate_unit_keys = [
            key
            for key, case_pass_rate in mandatory_gate_unit_case_pass_rates.items()
            if float(case_pass_rate) < (1.0 - 1.0e-12)
        ]
    else:
        active_invalid_mandatory_gate_unit_keys = [
            key
            for key, case_pass_rate in mandatory_gate_unit_case_pass_rates.items()
            if float(case_pass_rate) + 1.0e-12 < mandatory_metric_valid_case_pass_min_effective
        ]
    active_invalid_mandatory_gate_unit_count = int(len(active_invalid_mandatory_gate_unit_keys))
    valid_mandatory_gate_unit_count_case_rate = int(
        max(mandatory_total_gate_unit_count - active_invalid_mandatory_gate_unit_count, 0)
    )
    mandatory_gate_unit_evaluable_case_rates: dict[str, float] = {}
    for gate_unit_key, metric_keys_for_unit in mandatory_gate_unit_to_metric_keys.items():
        if not metric_keys_for_unit:
            mandatory_gate_unit_evaluable_case_rates[str(gate_unit_key)] = 0.0
            continue
        evaluable_case_count = 0
        for case_id in case_ids:
            if any(
                math.isfinite(float(per_case_metric_rel.get(case_id, {}).get(str(metric_key), float("nan"))))
                for metric_key in metric_keys_for_unit
            ):
                evaluable_case_count += 1
        mandatory_gate_unit_evaluable_case_rates[str(gate_unit_key)] = float(evaluable_case_count) / float(
            max(len(case_ids), 1)
        )
    valid_mandatory_gate_unit_count_coverage = int(
        sum(
            1
            for _, evaluable_case_rate in mandatory_gate_unit_evaluable_case_rates.items()
            if float(evaluable_case_rate) > 0.0
        )
    )
    mandatory_gate_unit_valid_count_shadow_evaluable_ratio = int(
        sum(
            1
            for _, evaluable_case_rate in mandatory_gate_unit_evaluable_case_rates.items()
            if float(evaluable_case_rate) + 1.0e-12
            >= mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective
        )
    )
    valid_mandatory_gate_unit_count = int(valid_mandatory_gate_unit_count_coverage)
    mandatory_all_gate_unit_keys = [str(k) for k in sorted(mandatory_gate_unit_case_pass_rates.keys())]
    mandatory_valid_gate_unit_key_set = set(mandatory_all_gate_unit_keys) - {
        str(k) for k in active_invalid_mandatory_gate_unit_keys
    }
    if mandatory_quality_hard_scope == "all_units":
        mandatory_quality_gate_unit_keys = list(mandatory_all_gate_unit_keys)
    else:
        mandatory_quality_gate_unit_keys = [
            key for key in mandatory_all_gate_unit_keys if key in mandatory_valid_gate_unit_key_set
        ]
    mandatory_quality_gate_unit_key_set = set(mandatory_quality_gate_unit_keys)
    mandatory_quality_metric_key_set: set[str] = set()
    for key in mandatory_quality_gate_unit_keys:
        mandatory_quality_metric_key_set.update(
            str(metric_key) for metric_key in list(mandatory_gate_unit_to_metric_keys.get(key, []))
        )
    mandatory_quality_metric_keys = [
        key for key in mandatory_gate_metric_keys if str(key) in mandatory_quality_metric_key_set
    ]
    mandatory_quality_gate_unit_count = int(len(mandatory_quality_gate_unit_keys))
    mandatory_quality_metric_count = int(len(mandatory_quality_metric_keys))

    def _compute_mandatory_case_rates(unit_keys: list[str]) -> tuple[float, float]:
        if not unit_keys:
            return 1.0, 1.0
        if mandatory_case_unit_weight_mode_effective == "family_weighted":
            total_weight = float(sum(float(mandatory_gate_unit_weights.get(key, 1.0)) for key in unit_keys))
            ratio_sum = 0.0
            for case_id in case_ids:
                passed_weight = 0.0
                for key in unit_keys:
                    if bool(mandatory_gate_unit_case_pass.get(key, {}).get(case_id, False)):
                        passed_weight += float(mandatory_gate_unit_weights.get(key, 1.0))
                ratio_sum += float(passed_weight) / float(max(total_weight, 1.0e-12))
            ratio_mean = ratio_sum / float(len(case_ids)) if case_ids else 0.0
        else:
            ratio_sum = 0.0
            for case_id in case_ids:
                passed_count = sum(
                    1
                    for key in unit_keys
                    if bool(mandatory_gate_unit_case_pass.get(key, {}).get(case_id, False))
                )
                ratio_sum += float(passed_count) / float(max(len(unit_keys), 1))
            ratio_mean = ratio_sum / float(len(case_ids)) if case_ids else 0.0
        case_ok_count = int(
            sum(
                1
                for case_id in case_ids
                if all(bool(mandatory_gate_unit_case_pass.get(key, {}).get(case_id, False)) for key in unit_keys)
            )
        )
        all_required = float(case_ok_count) / float(len(case_ids)) if case_ids else 0.0
        return float(ratio_mean), float(all_required)

    low_signal_optional_metric_keys = [
        key
        for key in sorted(optional_metric_key_set)
        if int(metric_case_counts.get(key, 0)) > 0
        and (
            float(metric_low_signal_counts.get(key, 0)) / float(max(metric_case_counts.get(key, 1), 1))
        )
        >= low_signal_case_ratio_threshold
    ]
    low_signal_optional_metric_key_set = set(low_signal_optional_metric_keys)
    optional_hard_metric_key_set = set(optional_metric_key_set) - low_signal_optional_metric_key_set
    suppressed_low_signal_metric_count = (
        len(low_signal_optional_metric_keys) if low_signal_policy == "suppress" else 0
    )
    effective_metric_count = int(len(mandatory_metric_key_set) + len(optional_hard_metric_key_set))

    mandatory_hard_mode = str(mandatory_cfg.get("mandatory_hard_mode", "hard_block_if_invalid")).strip().lower()
    if mandatory_hard_mode == "min_valid_count":
        min_valid_abs = max(int(mandatory_cfg.get("min_valid_mandatory_count_abs", 1) or 1), 0)
        min_valid_ratio = max(float(mandatory_cfg.get("min_valid_mandatory_ratio", 1.0) or 1.0), 0.0)
        min_valid_ratio_count = int(math.floor((min_valid_ratio * float(mandatory_total_gate_unit_count)) + 1.0e-12))
        min_valid_mandatory_count_effective = int(max(min_valid_abs, min_valid_ratio_count))
        if bool(mandatory_cfg.get("min_valid_mandatory_cap_by_total", True)) and mandatory_total_gate_unit_count > 0:
            min_valid_mandatory_count_effective = min(
                min_valid_mandatory_count_effective, mandatory_total_gate_unit_count
            )
        if mandatory_total_gate_unit_count <= 0:
            min_valid_mandatory_count_effective = 0
        mandatory_validity_passed = bool(
            valid_mandatory_gate_unit_count >= min_valid_mandatory_count_effective
        )
    else:
        min_valid_mandatory_count_effective = int(mandatory_total_gate_unit_count)
        mandatory_validity_passed = bool(
            mandatory_total_gate_unit_count <= 0
            or valid_mandatory_gate_unit_count >= mandatory_total_gate_unit_count
        )

    pass_rate_mandatory_case_ratio_mean_all_units, pass_rate_mandatory_case_all_required_all_units = (
        _compute_mandatory_case_rates(mandatory_all_gate_unit_keys)
    )
    pass_rate_mandatory_case_ratio_mean, pass_rate_mandatory_case_all_required = (
        _compute_mandatory_case_rates(mandatory_quality_gate_unit_keys)
    )

    mandatory_case_mode_effective = str(error_aggregation.get("mandatory_case_mode", "ratio_mean")).strip().lower()
    if mandatory_case_mode_effective not in {"ratio_mean", "all_required"}:
        mandatory_case_mode_effective = "ratio_mean"
    if mandatory_case_mode_effective == "all_required":
        pass_rate_mandatory_case = float(pass_rate_mandatory_case_all_required)
        pass_rate_mandatory_case_all_units = float(pass_rate_mandatory_case_all_required_all_units)
    else:
        pass_rate_mandatory_case = float(pass_rate_mandatory_case_ratio_mean)
        pass_rate_mandatory_case_all_units = float(pass_rate_mandatory_case_ratio_mean_all_units)
    mandatory_mean_aggregation_effective = str(error_aggregation.get("mandatory_mean_aggregation", "raw")).strip().lower()
    if mandatory_mean_aggregation_effective not in {"raw", "family_weighted"}:
        mandatory_mean_aggregation_effective = "raw"
    mandatory_mean_mode_effective = str(error_aggregation.get("mandatory_mean_mode", "winsorized")).strip().lower()
    if mandatory_mean_mode_effective not in {"winsorized", "raw"}:
        mandatory_mean_mode_effective = "winsorized"
    if optional_hard_metric_key_set:
        optional_case_ok_count = int(
            sum(
                1
                for case_id in case_ids
                if all(bool(per_case_metric_pass.get(case_id, {}).get(key, False)) for key in optional_hard_metric_key_set)
            )
        )
    else:
        optional_case_ok_count = int(len(case_ids))
    pass_rate_optional_case = (
        float(optional_case_ok_count) / float(len(case_ids)) if case_ids else 0.0
    )

    optional_metric_pass_rates: list[float] = []
    for key in sorted(optional_hard_metric_key_set):
        case_count = int(metric_case_counts.get(key, 0))
        pass_count = int(metric_pass_counts.get(key, 0))
        optional_metric_pass_rates.append(float(pass_count) / float(max(case_count, 1)))
    pass_rate_optional_metric_mean = (
        float(sum(optional_metric_pass_rates) / len(optional_metric_pass_rates))
        if optional_metric_pass_rates
        else 1.0
    )

    mandatory_rel_values_all_units: list[float] = []
    for key in mandatory_metric_key_set:
        mandatory_rel_values_all_units.extend(list(metric_rel_values.get(key) or []))
    mandatory_rel_values: list[float] = []
    for key in mandatory_quality_metric_keys:
        mandatory_rel_values.extend(list(metric_rel_values.get(key) or []))
    mean_rel_diff_mandatory_raw = (
        float(sum(mandatory_rel_values) / len(mandatory_rel_values))
        if mandatory_rel_values
        else None
    )
    mean_rel_diff_mandatory_all_units = (
        float(sum(mandatory_rel_values_all_units) / len(mandatory_rel_values_all_units))
        if mandatory_rel_values_all_units
        else None
    )
    mandatory_family_rel_values: dict[str, list[float]] = {}
    for key in mandatory_quality_metric_keys:
        key_rel_values = list(metric_rel_values.get(key) or [])
        if not key_rel_values:
            continue
        family = _metric_family(key, metric_taxonomy)
        mandatory_family_rel_values.setdefault(family, []).extend(key_rel_values)
    mean_rel_diff_mandatory_family_weighted: float | None = None
    if mandatory_family_rel_values:
        weighted_sum = 0.0
        weight_total = 0.0
        for family, values in mandatory_family_rel_values.items():
            family_mean = float(sum(values) / len(values))
            family_weight = _metric_family_weight(family, mandatory_family_weights_cfg)
            weighted_sum += family_weight * family_mean
            weight_total += family_weight
        if weight_total > 0.0:
            mean_rel_diff_mandatory_family_weighted = float(weighted_sum / weight_total)

    optional_rel_values: list[float] = []
    for key in optional_hard_metric_key_set:
        optional_rel_values.extend(list(metric_rel_values.get(key) or []))
    mean_rel_diff_optional = (
        float(sum(optional_rel_values) / len(optional_rel_values))
        if optional_rel_values
        else None
    )

    optional_metric_pass_min = float(error_aggregation.get("optional_metric_pass_min", mandatory_case_pass_min) or mandatory_case_pass_min)
    max_mean_rel_diff_mandatory = float(error_aggregation.get("max_mean_rel_diff_mandatory", rel_tolerance) or rel_tolerance)
    max_mean_rel_diff_optional = float(error_aggregation.get("max_mean_rel_diff_optional", rel_tolerance) or rel_tolerance)
    mandatory_winsor_cap_multiplier = float(error_aggregation.get("mandatory_winsor_cap_multiplier", 3.0) or 3.0)
    mandatory_outlier_multiplier = float(error_aggregation.get("mandatory_outlier_multiplier", 5.0) or 5.0)
    mandatory_outlier_ratio_max_effective = float(
        np.clip(float(error_aggregation.get("mandatory_outlier_ratio_max", 0.20) or 0.20), 0.0, 1.0)
    )
    mandatory_tail_guard_mode_effective = str(
        error_aggregation.get("mandatory_tail_guard_mode", "p95")
    ).strip().lower()
    if mandatory_tail_guard_mode_effective not in {"p95", "none"}:
        mandatory_tail_guard_mode_effective = "p95"
    mandatory_tail_guard_policy_effective = str(
        error_aggregation.get("mandatory_tail_guard_policy", "conditional_hard")
    ).strip().lower()
    if mandatory_tail_guard_policy_effective not in {"conditional_hard", "hard", "score_only"}:
        mandatory_tail_guard_policy_effective = "conditional_hard"
    mandatory_tail_activation_ratio_min_effective = float(
        np.clip(float(error_aggregation.get("mandatory_tail_activation_ratio_min", 0.10) or 0.10), 0.0, 1.0)
    )
    mandatory_tail_exceed_ref_effective = str(
        error_aggregation.get("mandatory_tail_exceed_ref", "tail_max")
    ).strip().lower()
    if mandatory_tail_exceed_ref_effective not in {"tail_max", "mean_cap"}:
        mandatory_tail_exceed_ref_effective = "tail_max"
    mandatory_tail_rel_diff_max_effective = float(
        max(0.0, float(error_aggregation.get("mandatory_tail_rel_diff_max", 1.50) or 1.50))
    )
    mandatory_tail_min_samples = int(max(1, int(error_aggregation.get("mandatory_tail_min_samples", 8) or 8)))
    mandatory_error_include_validity_effective = bool(
        error_aggregation.get("mandatory_error_include_validity", False)
    )
    winsor_cap = float(max(0.0, mandatory_winsor_cap_multiplier) * max(max_mean_rel_diff_mandatory, 0.0))
    mean_rel_diff_mandatory_winsorized: float | None = None
    if mandatory_rel_values:
        if winsor_cap > 0.0 and math.isfinite(winsor_cap):
            mandatory_rel_values_winsorized = [float(min(v, winsor_cap)) for v in mandatory_rel_values]
        else:
            mandatory_rel_values_winsorized = list(mandatory_rel_values)
        mean_rel_diff_mandatory_winsorized = float(
            sum(mandatory_rel_values_winsorized) / len(mandatory_rel_values_winsorized)
        )
    outlier_threshold = float(max(0.0, mandatory_outlier_multiplier) * max(max_mean_rel_diff_mandatory, 0.0))
    if mandatory_rel_values and outlier_threshold > 0.0:
        mandatory_rel_outlier_ratio = float(
            sum(1 for v in mandatory_rel_values if float(v) > outlier_threshold)
        ) / float(len(mandatory_rel_values))
    else:
        mandatory_rel_outlier_ratio = 0.0
    if mandatory_rel_values_all_units and outlier_threshold > 0.0:
        mandatory_rel_outlier_ratio_all_units = float(
            sum(1 for v in mandatory_rel_values_all_units if float(v) > outlier_threshold)
        ) / float(len(mandatory_rel_values_all_units))
    else:
        mandatory_rel_outlier_ratio_all_units = 0.0
    mandatory_rel_diff_p95_all_units: float | None = None
    if mandatory_rel_values_all_units:
        mandatory_rel_diff_p95_all_units = float(
            np.quantile(np.asarray(mandatory_rel_values_all_units, dtype=float), 0.95)
        )
    tail_rel_values = (
        list(mandatory_rel_values_all_units)
        if mandatory_tail_scope_effective == "all_units"
        else list(mandatory_rel_values)
    )
    mandatory_rel_diff_p95: float | None = None
    if tail_rel_values:
        mandatory_rel_diff_p95 = float(np.quantile(np.asarray(tail_rel_values, dtype=float), 0.95))
    mandatory_tail_guard_triggered = bool(
        mandatory_rel_diff_p95 is not None and mandatory_rel_diff_p95 > mandatory_tail_rel_diff_max_effective
    )
    if mandatory_tail_exceed_ref_effective == "mean_cap":
        tail_exceed_ref = float(max(max_mean_rel_diff_mandatory, 0.0))
    else:
        tail_exceed_ref = float(max(mandatory_tail_rel_diff_max_effective, 0.0))
    if tail_rel_values and tail_exceed_ref > 0.0:
        mandatory_tail_exceed_ratio = float(
            sum(1 for v in tail_rel_values if float(v) > tail_exceed_ref)
        ) / float(len(tail_rel_values))
    else:
        mandatory_tail_exceed_ratio = 0.0
    tail_guard_sample_eligible = bool(
        mandatory_rel_diff_p95 is not None and len(tail_rel_values) >= mandatory_tail_min_samples
    )
    if mandatory_tail_guard_mode_effective == "none":
        mandatory_tail_guard_hard_applied = False
    elif mandatory_tail_guard_policy_effective == "score_only":
        mandatory_tail_guard_hard_applied = False
    elif mandatory_tail_guard_policy_effective == "hard":
        mandatory_tail_guard_hard_applied = bool(tail_guard_sample_eligible and mandatory_tail_guard_triggered)
    else:
        mandatory_tail_guard_hard_applied = bool(
            tail_guard_sample_eligible
            and mandatory_tail_guard_triggered
            and mandatory_tail_exceed_ratio + 1.0e-12 >= mandatory_tail_activation_ratio_min_effective
        )
    mandatory_tail_guard_passed = bool(not mandatory_tail_guard_hard_applied)
    if mandatory_mean_mode_effective == "winsorized":
        mean_rel_diff_mandatory = (
            mean_rel_diff_mandatory_winsorized
            if mean_rel_diff_mandatory_winsorized is not None
            else mean_rel_diff_mandatory_raw
        )
    elif (
        mandatory_mean_aggregation_effective == "family_weighted"
        and mean_rel_diff_mandatory_family_weighted is not None
    ):
        mean_rel_diff_mandatory = float(mean_rel_diff_mandatory_family_weighted)
    else:
        mean_rel_diff_mandatory = mean_rel_diff_mandatory_raw
    optional_weight = float(np.clip(float(error_aggregation.get("optional_weight", 0.35) or 0.35), 0.0, 1.0))
    mandatory_quality_scope_empty = bool(mandatory_quality_metric_count <= 0)

    mandatory_quality_passed = bool(
        (
            mandatory_quality_scope_empty
            or pass_rate_mandatory_case >= mandatory_case_pass_min
        )
        and (
            mean_rel_diff_mandatory is None
            or float(mean_rel_diff_mandatory) <= max_mean_rel_diff_mandatory
        )
        and float(mandatory_rel_outlier_ratio) <= mandatory_outlier_ratio_max_effective
        and bool(mandatory_tail_guard_passed)
    )
    optional_quality_passed = bool(
        pass_rate_optional_metric_mean >= optional_metric_pass_min
        and (
            mean_rel_diff_optional is None
            or float(mean_rel_diff_optional) <= max_mean_rel_diff_optional
        )
    )
    coverage_gate_passed = bool(mandatory_validity_passed)
    if mandatory_error_include_validity_effective:
        mandatory_error_passed = bool(coverage_gate_passed and mandatory_quality_passed)
    else:
        mandatory_error_passed = bool(mandatory_quality_passed)
    optional_error_passed = bool(optional_quality_passed)
    error_gate_passed = bool(mandatory_error_passed and optional_error_passed)

    error_gate_passed, error_fail_reason_primary = _compose_error_gate(
        error_aggregation_mode=error_aggregation_mode,
        mandatory_error_passed=mandatory_error_passed,
        optional_error_passed=optional_error_passed,
        mandatory_quality_scope_empty=mandatory_quality_scope_empty,
        pass_rate_mandatory_case=pass_rate_mandatory_case,
        mandatory_case_pass_min=mandatory_case_pass_min,
        mean_rel_diff_mandatory=mean_rel_diff_mandatory,
        max_mean_rel_diff_mandatory=max_mean_rel_diff_mandatory,
        mandatory_rel_outlier_ratio=mandatory_rel_outlier_ratio,
        mandatory_outlier_ratio_max_effective=mandatory_outlier_ratio_max_effective,
        mandatory_tail_guard_passed=mandatory_tail_guard_passed,
    )

    mandatory_weight = 1.0 - optional_weight
    if mandatory_quality_scope_effective == "hybrid_score":
        mandatory_pass_rate_for_score = float(pass_rate_mandatory_case_all_units)
    else:
        mandatory_pass_rate_for_score = float(pass_rate_mandatory_case)
    error_gate_score = float(
        mandatory_weight * mandatory_pass_rate_for_score + optional_weight * pass_rate_optional_metric_mean
    )
    evaluation_contract_version = str(policy_cfg.get("evaluation_contract_version") or "")
    metric_taxonomy_profile_effective = str(
        metric_taxonomy.get("profile", policy_cfg.get("metric_taxonomy_profile_effective", "legacy_builtin"))
    )
    diagnostic_schema_ok = bool(policy_cfg.get("diagnostic_schema_ok", True))

    summary = {
        "cases": len(case_ids),
        "pass_rate": pass_rate,
        "pass_rate_all_metric_legacy": pass_rate,
        "pass_cases": pass_cases,
        "failed_cases": failed_cases,
        "qoi_metrics_count": len(metric_keys),
        "max_rel_diff": max(rel_values) if rel_values else None,
        "mean_rel_diff": (sum(rel_values) / len(rel_values)) if rel_values else None,
        "mean_rel_diff_all_metric_legacy": (sum(rel_values) / len(rel_values)) if rel_values else None,
        "worst_case": worst_row,
        "rel_tolerance": rel_tolerance,
        "rel_eps": rel_eps,
        "mandatory_total_metric_count": int(mandatory_total_metric_count),
        "valid_mandatory_metric_count": int(valid_mandatory_metric_count),
        "mandatory_total_gate_unit_count": int(mandatory_total_gate_unit_count),
        "valid_mandatory_gate_unit_count": int(valid_mandatory_gate_unit_count),
        "valid_mandatory_gate_unit_count_case_rate": int(valid_mandatory_gate_unit_count_case_rate),
        "valid_mandatory_gate_unit_count_coverage": int(valid_mandatory_gate_unit_count_coverage),
        "mandatory_validity_basis_effective": str(mandatory_validity_basis_effective),
        "invalid_mandatory_metric_count": int(invalid_mandatory_metric_count),
        "inactive_mandatory_metric_count": int(inactive_mandatory_metric_count),
        "active_invalid_mandatory_metric_count": int(active_invalid_mandatory_metric_count),
        "active_invalid_mandatory_gate_unit_keys": [str(k) for k in active_invalid_mandatory_gate_unit_keys],
        "min_valid_mandatory_count_effective": int(min_valid_mandatory_count_effective),
        "mandatory_validity_passed": bool(mandatory_validity_passed),
        "mandatory_metric_failures": {k: int(v) for k, v in mandatory_fail_cases.items()},
        "mandatory_metric_case_pass_rates": {k: float(v) for k, v in mandatory_metric_case_pass_rates.items()},
        "mandatory_gate_unit_case_pass_rates": {
            str(k): float(v) for k, v in mandatory_gate_unit_case_pass_rates.items()
        },
        "mandatory_gate_unit_evaluable_case_rates": {
            str(k): float(v) for k, v in mandatory_gate_unit_evaluable_case_rates.items()
        },
        "mandatory_gate_unit_valid_count_shadow_evaluable_ratio": int(
            mandatory_gate_unit_valid_count_shadow_evaluable_ratio
        ),
        "mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective": float(
            mandatory_gate_unit_min_evaluable_case_ratio_shadow_effective
        ),
        "mandatory_gate_unit_mode_effective": str(mandatory_valid_unit_mode_effective),
        "mandatory_species_family_score_mode_effective": str(
            mandatory_species_family_score_mode_effective
        ),
        "mandatory_quality_scope_effective": str(mandatory_quality_scope_effective),
        "mandatory_tail_scope_effective": str(mandatory_tail_scope_effective),
        "mandatory_case_unit_weight_mode_effective": str(
            mandatory_case_unit_weight_mode_effective
        ),
        "mandatory_quality_gate_unit_count": int(mandatory_quality_gate_unit_count),
        "mandatory_quality_metric_count": int(mandatory_quality_metric_count),
        "mandatory_species_family_case_pass_min_effective": float(
            mandatory_species_family_case_pass_min_effective
        ),
        "mandatory_metric_validity_mode_effective": str(mandatory_metric_validity_mode_effective),
        "mandatory_metric_valid_case_pass_min_effective": float(mandatory_metric_valid_case_pass_min_effective),
        "pass_rate_mandatory_case": float(pass_rate_mandatory_case),
        "pass_rate_mandatory_case_all_units": float(pass_rate_mandatory_case_all_units),
        "pass_rate_mandatory_case_all_required": float(pass_rate_mandatory_case_all_required),
        "pass_rate_mandatory_case_ratio_mean": float(pass_rate_mandatory_case_ratio_mean),
        "pass_rate_mandatory_case_all_required_all_units": float(
            pass_rate_mandatory_case_all_required_all_units
        ),
        "pass_rate_mandatory_case_ratio_mean_all_units": float(
            pass_rate_mandatory_case_ratio_mean_all_units
        ),
        "pass_rate_optional_case": float(pass_rate_optional_case),
        "pass_rate_optional_metric_mean": float(pass_rate_optional_metric_mean),
        "mean_rel_diff_mandatory": mean_rel_diff_mandatory,
        "mean_rel_diff_mandatory_all_units": mean_rel_diff_mandatory_all_units,
        "mean_rel_diff_mandatory_raw": mean_rel_diff_mandatory_raw,
        "mean_rel_diff_mandatory_family_weighted": mean_rel_diff_mandatory_family_weighted,
        "mean_rel_diff_mandatory_winsorized": mean_rel_diff_mandatory_winsorized,
        "mean_rel_diff_optional": mean_rel_diff_optional,
        "mandatory_rel_outlier_ratio": float(mandatory_rel_outlier_ratio),
        "mandatory_rel_outlier_ratio_all_units": float(mandatory_rel_outlier_ratio_all_units),
        "mandatory_rel_outlier_ratio_max_effective": float(mandatory_outlier_ratio_max_effective),
        "mandatory_rel_diff_p95": mandatory_rel_diff_p95,
        "mandatory_rel_diff_p95_all_units": mandatory_rel_diff_p95_all_units,
        "mandatory_tail_guard_passed": bool(mandatory_tail_guard_passed),
        "mandatory_tail_guard_triggered": bool(mandatory_tail_guard_triggered),
        "mandatory_tail_guard_hard_applied": bool(mandatory_tail_guard_hard_applied),
        "mandatory_tail_guard_mode_effective": str(mandatory_tail_guard_mode_effective),
        "mandatory_tail_guard_policy_effective": str(mandatory_tail_guard_policy_effective),
        "mandatory_tail_activation_ratio_min_effective": float(mandatory_tail_activation_ratio_min_effective),
        "mandatory_tail_exceed_ref_effective": str(mandatory_tail_exceed_ref_effective),
        "mandatory_tail_exceed_ratio": float(mandatory_tail_exceed_ratio),
        "mandatory_tail_rel_diff_max_effective": float(mandatory_tail_rel_diff_max_effective),
        "mandatory_quality_scope_empty": bool(mandatory_quality_scope_empty),
        "error_gate_score": float(error_gate_score),
        "error_gate_passed": bool(error_gate_passed),
        "coverage_gate_passed": bool(coverage_gate_passed),
        "mandatory_quality_passed": bool(mandatory_quality_passed),
        "optional_quality_passed": bool(optional_quality_passed),
        "mandatory_error_passed": bool(mandatory_error_passed),
        "optional_error_passed": bool(optional_error_passed),
        "error_fail_reason_primary": str(error_fail_reason_primary),
        "mandatory_error_include_validity_effective": bool(mandatory_error_include_validity_effective),
        "mandatory_case_pass_min": float(mandatory_case_pass_min),
        "mandatory_case_mode_effective": str(mandatory_case_mode_effective),
        "mandatory_mean_aggregation_effective": str(mandatory_mean_aggregation_effective),
        "mandatory_mean_mode_effective": str(mandatory_mean_mode_effective),
        "optional_metric_pass_min": float(optional_metric_pass_min),
        "max_mean_rel_diff_mandatory": float(max_mean_rel_diff_mandatory),
        "max_mean_rel_diff_optional": float(max_mean_rel_diff_optional),
        "error_aggregation_mode": error_aggregation_mode,
        "optional_metric_count": int(len(optional_metric_key_set)),
        "optional_hard_metric_count": int(len(optional_hard_metric_key_set)),
        "effective_metric_count": int(effective_metric_count),
        "suppressed_low_signal_metric_count": int(suppressed_low_signal_metric_count),
        "low_signal_optional_metrics": low_signal_optional_metric_keys,
        "low_signal_policy": low_signal_policy,
        "denominator_mode": denominator_mode,
        "evaluation_contract_version": evaluation_contract_version,
        "metric_taxonomy_profile_effective": metric_taxonomy_profile_effective,
        "diagnostic_schema_ok": bool(diagnostic_schema_ok),
    }
    return comparison, summary


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_failure_reason(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return "missing_file"
    if isinstance(exc, (csv.Error, UnicodeDecodeError)):
        return "invalid_csv"
    if ct is not None and isinstance(exc, ct.CanteraError):
        msg = str(exc).lower()
        if "phase" in msg:
            return "invalid_phase"
        return "cantera_load_error"
    if isinstance(exc, ValueError):
        msg = str(exc).lower()
        if "conditions" in msg or "csv" in msg:
            return "invalid_csv"
        return "invalid_value"
    return "runtime_error"


def _write_summary_outputs(out_dir: Path, summary: dict[str, Any], cfg: dict[str, Any] | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg is not None:
        (out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


def _apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(cfg)

    if args.candidate_mechanism is not None:
        cand = dict(out.get("candidate") or {})
        cand["mechanism"] = args.candidate_mechanism
        out["candidate"] = cand

    if args.conditions_csv is not None:
        out["conditions_csv"] = args.conditions_csv

    if args.rel_tolerance is not None:
        eval_cfg = dict(out.get("evaluation") or {})
        eval_cfg["rel_tolerance"] = float(args.rel_tolerance)
        out["evaluation"] = eval_cfg

    if args.n_steps is not None:
        integ = dict(out.get("integration") or {})
        integ["n_steps"] = int(args.n_steps)
        out["integration"] = integ

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Cantera runner/evaluator")
    parser.add_argument("--config", required=True, help="Path to run config YAML")
    parser.add_argument("--run-id", default=None, help="Optional run id override")
    parser.add_argument(
        "--candidate-mechanism",
        default=None,
        help="Override candidate mechanism path in config.",
    )
    parser.add_argument(
        "--conditions-csv",
        default=None,
        help="Override conditions CSV path in config.",
    )
    parser.add_argument(
        "--rel-tolerance",
        default=None,
        type=float,
        help="Override evaluation.rel_tolerance.",
    )
    parser.add_argument(
        "--n-steps",
        default=None,
        type=int,
        help="Override integration.n_steps.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    fallback_run_id = args.run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir: Path | None = None
    cfg: dict[str, Any] | None = None

    try:
        cfg = _apply_cli_overrides(load_yaml(config_path), args)

        run_id = args.run_id or str(cfg.get("run_id") or fallback_run_id)
        root_dir = (config_path.parent / str((cfg.get("output") or {}).get("root", "../runs"))).resolve()
        out_dir = root_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        simulation_cfg = cfg.get("simulation") or {}
        mode = str(simulation_cfg.get("mode", "gas_homogeneous"))

        baseline_cfg = cfg.get("baseline") or {}
        candidate_cfg = cfg.get("candidate") or {}

        baseline_mech = _resolve_mechanism_ref(
            baseline_cfg.get("mechanism", "assets/mechanisms/gri30.yaml"),
            config_parent=config_path.parent,
        )
        candidate_mech_raw = candidate_cfg.get("mechanism")
        candidate_mech = (
            _resolve_mechanism_ref(candidate_mech_raw, config_parent=config_path.parent)
            if candidate_mech_raw
            else None
        )

        phase = _phase_name_from_config(cfg)
        surface_cfg = _surface_cfg(cfg)
        schema_cfg = _surface_schema_cfg(cfg) if mode == "surface_batch" else {}

        conditions_csv = _resolve_input_path(
            cfg.get("conditions_csv", "assets/conditions/gri30_small.csv"),
            config_parent=config_path.parent,
        )

        _preflight(
            baseline_mech=baseline_mech,
            candidate_mech=candidate_mech,
            phase=phase,
            conditions_csv=conditions_csv,
            mode=mode,
            surface_cfg=surface_cfg,
            schema_cfg=schema_cfg,
        )
        conditions = load_conditions(conditions_csv, mode=mode, schema_cfg=schema_cfg)

        mix = cfg.get("mixture") or {}
        fuel = str(mix.get("fuel", "CH4:1.0"))
        oxidizer = str(mix.get("oxidizer", "O2:1.0, N2:3.76"))

        integ = cfg.get("integration") or {}
        n_steps = int(integ.get("n_steps", 400))
        if n_steps < 2:
            raise ValueError("integration.n_steps must be >= 2")

        qoi = cfg.get("qoi") or {}
        species_last = list(qoi.get("species_last") or ["CO2", "CO", "H2O", "CH4", "O2", "NO", "NO2"])
        species_max = list(qoi.get("species_max") or ["OH", "HO2"])
        surface_qoi_cfg = _surface_qoi_cfg(cfg)
        if mode == "surface_batch" and not list(surface_qoi_cfg.get("selectors") or []):
            raise ValueError("evaluation.qoi.selectors is required for surface mode")

        eval_cfg = cfg.get("evaluation") or {}
        rel_tolerance = float(eval_cfg.get("rel_tolerance", 0.2))
        rel_eps = float(eval_cfg.get("rel_eps", 1.0e-12))
        qoi_builtin_metrics = dict(eval_cfg.get("qoi_builtin_metrics") or {})
        eval_policy = {
            "error_aggregation": dict(eval_cfg.get("error_aggregation") or {}),
            "metric_normalization": dict(eval_cfg.get("metric_normalization") or {}),
            "metric_taxonomy": dict(eval_cfg.get("metric_taxonomy_resolved") or eval_cfg.get("metric_taxonomy") or {}),
            "evaluation_contract_version": str(
                ((eval_cfg.get("contract") or {}).get("version") or "")
            ),
            "metric_taxonomy_profile_effective": str(
                ((eval_cfg.get("metric_taxonomy") or {}).get("profile") or "legacy_builtin")
            ),
            "diagnostic_schema_ok": True,
        }
        if rel_tolerance < 0.0:
            raise ValueError("evaluation.rel_tolerance must be >= 0")
        if rel_eps <= 0.0:
            raise ValueError("evaluation.rel_eps must be > 0")
        gas_qoi_runtime_cfg = {"qoi_builtin_metrics": qoi_builtin_metrics}

        baseline_rows = run_mechanism(
            baseline_mech,
            phase,
            conditions,
            fuel=fuel,
            oxidizer=oxidizer,
            n_steps=n_steps,
            species_last=species_last,
            species_max=species_max,
            mode=mode,
            surface_cfg=surface_cfg,
            qoi_cfg=(surface_qoi_cfg if mode == "surface_batch" else gas_qoi_runtime_cfg),
        )
        write_rows_csv(out_dir / "baseline_results.csv", baseline_rows)

        summary: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "ok",
            "failure_reason": None,
            "run_id": run_id,
            "config": str(config_path),
            "conditions_csv": str(conditions_csv),
            "simulation_mode": mode,
            "phase": phase if mode == "gas_homogeneous" else str(surface_cfg.get("interface_phase")),
            "baseline_mechanism": str(baseline_mech),
            "baseline_counts": mechanism_counts(
                baseline_mech,
                phase,
                mode=mode,
                gas_phase=str(surface_cfg.get("gas_phase", "gas")),
                interface_phase=str(surface_cfg.get("interface_phase") or ""),
            ),
            "qoi_metrics_count": 0,
            "pass_cases": None,
            "failed_cases": None,
            "worst_case": None,
        }

        if mode == "surface_batch":
            summary["gas_phase"] = str(surface_cfg.get("gas_phase", "gas"))
            summary["surface_phase"] = str(surface_cfg.get("interface_phase") or "")

        if candidate_mech is not None:
            candidate_rows = run_mechanism(
                candidate_mech,
                phase,
                conditions,
                fuel=fuel,
                oxidizer=oxidizer,
                n_steps=n_steps,
                species_last=species_last,
                species_max=species_max,
                mode=mode,
                surface_cfg=surface_cfg,
                qoi_cfg=(surface_qoi_cfg if mode == "surface_batch" else gas_qoi_runtime_cfg),
            )
            write_rows_csv(out_dir / "candidate_results.csv", candidate_rows)

            comparison_rows, comp_summary = compare_rows(
                baseline_rows,
                candidate_rows,
                rel_eps=rel_eps,
                rel_tolerance=rel_tolerance,
                eval_policy=eval_policy,
            )
        else:
            candidate_rows = list(baseline_rows)
            write_rows_csv(out_dir / "candidate_results.csv", candidate_rows)
            comparison_rows, comp_summary = compare_rows(
                baseline_rows,
                candidate_rows,
                rel_eps=rel_eps,
                rel_tolerance=rel_tolerance,
                eval_policy=eval_policy,
            )

        write_rows_csv(out_dir / "comparison_results.csv", comparison_rows)

        summary["candidate_mechanism"] = str(candidate_mech) if candidate_mech is not None else None
        summary["candidate_counts"] = (
            mechanism_counts(
                candidate_mech,
                phase,
                mode=mode,
                gas_phase=str(surface_cfg.get("gas_phase", "gas")),
                interface_phase=str(surface_cfg.get("interface_phase") or ""),
            )
            if candidate_mech is not None
            else summary["baseline_counts"]
        )
        summary["comparison"] = comp_summary
        summary["qoi_metrics_count"] = int(comp_summary.get("qoi_metrics_count") or 0)
        summary["pass_cases"] = int(comp_summary.get("pass_cases") or 0)
        summary["failed_cases"] = int(comp_summary.get("failed_cases") or 0)
        summary["worst_case"] = comp_summary.get("worst_case")

        _write_summary_outputs(out_dir, summary, cfg)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:
        if out_dir is None:
            out_dir = (config_path.parent / "../runs").resolve() / fallback_run_id
        failure_summary: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "failure_reason": _resolve_failure_reason(exc),
            "error_message": str(exc),
            "run_id": (args.run_id or fallback_run_id),
            "config": str(config_path),
            "qoi_metrics_count": 0,
            "pass_cases": 0,
            "failed_cases": 0,
            "worst_case": None,
        }
        _write_summary_outputs(out_dir, failure_summary, cfg)
        print(json.dumps(failure_summary, ensure_ascii=False, indent=2))
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
