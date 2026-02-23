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

    out: dict[str, float] = {
        "ignition_delay": _ignition_delay(times, temps),
        "T_max": max_temp,
        "T_last": float(temps[-1]),
    }

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


def _safe_rel_diff(a: float, b: float, eps: float) -> float:
    if math.isnan(a) and math.isnan(b):
        return 0.0
    if math.isnan(a) or math.isnan(b):
        return float("inf")
    den = max(abs(a), eps)
    return abs(b - a) / den


def compare_rows(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    rel_eps: float,
    rel_tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_case_base = {str(r["case_id"]): r for r in baseline_rows}
    by_case_cand = {str(r["case_id"]): r for r in candidate_rows}
    case_ids = sorted(set(by_case_base) & set(by_case_cand))
    if not case_ids:
        raise ValueError("no overlapping cases between baseline and candidate")

    comparison: list[dict[str, Any]] = []
    per_case_pass: list[bool] = []
    rel_values: list[float] = []

    metric_keys = sorted(k for k in baseline_rows[0].keys() if k != "case_id")
    worst_row: dict[str, Any] | None = None
    worst_rel = -float("inf")
    for case_id in case_ids:
        b = by_case_base[case_id]
        c = by_case_cand[case_id]
        case_ok = True
        for key in metric_keys:
            b_val = float(b.get(key, float("nan")))
            c_val = float(c.get(key, float("nan")))
            rel = _safe_rel_diff(b_val, c_val, rel_eps)
            passed = bool(math.isfinite(rel) and rel <= rel_tolerance)
            if not passed:
                case_ok = False
            if math.isfinite(rel):
                rel_values.append(rel)
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
                }
            )
        per_case_pass.append(case_ok)

    pass_cases = int(sum(1 for ok in per_case_pass if ok))
    failed_cases = int(len(per_case_pass) - pass_cases)
    pass_rate = float(pass_cases) / float(len(per_case_pass))
    summary = {
        "cases": len(case_ids),
        "pass_rate": pass_rate,
        "pass_cases": pass_cases,
        "failed_cases": failed_cases,
        "qoi_metrics_count": len(metric_keys),
        "max_rel_diff": max(rel_values) if rel_values else None,
        "mean_rel_diff": (sum(rel_values) / len(rel_values)) if rel_values else None,
        "worst_case": worst_row,
        "rel_tolerance": rel_tolerance,
        "rel_eps": rel_eps,
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
        if rel_tolerance < 0.0:
            raise ValueError("evaluation.rel_tolerance must be >= 0")
        if rel_eps <= 0.0:
            raise ValueError("evaluation.rel_eps must be > 0")

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
            qoi_cfg=surface_qoi_cfg,
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
                qoi_cfg=surface_qoi_cfg,
            )
            write_rows_csv(out_dir / "candidate_results.csv", candidate_rows)

            comparison_rows, comp_summary = compare_rows(
                baseline_rows,
                candidate_rows,
                rel_eps=rel_eps,
                rel_tolerance=rel_tolerance,
            )
        else:
            candidate_rows = list(baseline_rows)
            write_rows_csv(out_dir / "candidate_results.csv", candidate_rows)
            comparison_rows, comp_summary = compare_rows(
                baseline_rows,
                candidate_rows,
                rel_eps=rel_eps,
                rel_tolerance=rel_tolerance,
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
