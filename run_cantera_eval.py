#!/usr/bin/env python3
"""Standalone Cantera evaluator for GRI30-based benchmark conditions.

This tool is intentionally independent from rxn_platform internals.
It runs baseline/candidate mechanisms over condition CSVs and reports QoI diffs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cantera as ct
import yaml


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


def load_conditions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"T0", "P0_atm", "phi", "t_end"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"conditions missing required columns: {sorted(missing)}")
        for idx, row in enumerate(reader):
            case_id = (row.get("case_id") or "").strip() or f"row_{idx:04d}"
            rows.append(
                {
                    "case_id": case_id,
                    "T0": _as_float(row.get("T0"), "T0"),
                    "P0_atm": _as_float(row.get("P0_atm"), "P0_atm"),
                    "phi": _as_float(row.get("phi"), "phi"),
                    "t_end": _as_float(row.get("t_end"), "t_end"),
                }
            )
    if not rows:
        raise ValueError("conditions CSV has no rows")
    return rows


def mechanism_counts(mech_path: Path, phase: str) -> dict[str, int]:
    gas = ct.Solution(str(mech_path), phase)
    return {"species": int(gas.n_species), "reactions": int(gas.n_reactions)}


def _load_mechanism_or_raise(mech_path: Path, phase: str) -> ct.Solution:
    if not mech_path.exists():
        raise FileNotFoundError(f"mechanism file not found: {mech_path}")
    if not mech_path.is_file():
        raise ValueError(f"mechanism path is not a file: {mech_path}")
    return ct.Solution(str(mech_path), phase)


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


def _preflight(
    *,
    baseline_mech: Path,
    candidate_mech: Path | None,
    phase: str,
    conditions_csv: Path,
) -> None:
    _ = load_conditions(conditions_csv)
    _ = _load_mechanism_or_raise(baseline_mech, phase)
    if candidate_mech is not None:
        _ = _load_mechanism_or_raise(candidate_mech, phase)


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


def run_case(
    mech_path: Path,
    phase: str,
    case: dict[str, Any],
    *,
    fuel: str,
    oxidizer: str,
    n_steps: int,
    species_last: list[str],
    species_max: list[str],
) -> dict[str, float]:
    gas = ct.Solution(str(mech_path), phase)
    gas.set_equivalence_ratio(case["phi"], fuel, oxidizer)
    gas.TP = case["T0"], case["P0_atm"] * ct.one_atm

    reactor = ct.IdealGasConstPressureReactor(gas)
    net = ct.ReactorNet([reactor])

    t_end = max(case["t_end"], 1.0e-12)
    n_steps = max(int(n_steps), 2)

    times = [0.0]
    temps = [float(reactor.T)]
    max_temp = float(reactor.T)

    max_species: dict[str, float] = {}
    for sp in species_max:
        max_species[sp] = float(max(0.0, reactor.thermo[sp].X[0])) if sp in reactor.thermo.species_names else float("nan")

    for i in range(n_steps):
        target = t_end * float(i + 1) / float(n_steps)
        net.advance(target)
        t = float(net.time)
        temp = float(reactor.T)
        times.append(t)
        temps.append(temp)
        if temp > max_temp:
            max_temp = temp
        for sp in species_max:
            if sp not in reactor.thermo.species_names:
                continue
            val = float(max(0.0, reactor.thermo[sp].X[0]))
            if math.isnan(max_species.get(sp, float("nan"))):
                max_species[sp] = val
            elif val > max_species[sp]:
                max_species[sp] = val

    out: dict[str, float] = {
        "ignition_delay": _ignition_delay(times, temps),
        "T_max": max_temp,
        "T_last": float(reactor.T),
    }

    names = set(reactor.thermo.species_names)
    for sp in species_last:
        key = f"X_last:{sp}"
        out[key] = float(max(0.0, reactor.thermo[sp].X[0])) if sp in names else float("nan")
    for sp in species_max:
        key = f"X_max:{sp}"
        out[key] = float(max_species.get(sp, float("nan")))
    return out


def run_mechanism(
    mech_path: Path,
    phase: str,
    conditions: list[dict[str, Any]],
    *,
    fuel: str,
    oxidizer: str,
    n_steps: int,
    species_last: list[str],
    species_max: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in conditions:
        metrics = run_case(
            mech_path,
            phase,
            case,
            fuel=fuel,
            oxidizer=oxidizer,
            n_steps=n_steps,
            species_last=species_last,
            species_max=species_max,
        )
        row = {"case_id": case["case_id"], **metrics}
        rows.append(row)
    return rows


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
    if isinstance(exc, ct.CanteraError):
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
    parser = argparse.ArgumentParser(description="Standalone Cantera GRI30 runner/evaluator")
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

        baseline_cfg = cfg.get("baseline") or {}
        candidate_cfg = cfg.get("candidate") or {}

        baseline_mech = _resolve_input_path(
            baseline_cfg.get("mechanism", "assets/mechanisms/gri30.yaml"),
            config_parent=config_path.parent,
        )
        candidate_mech_raw = candidate_cfg.get("mechanism")
        candidate_mech = (
            _resolve_input_path(candidate_mech_raw, config_parent=config_path.parent)
            if candidate_mech_raw
            else None
        )
        phase = _phase_name_from_config(cfg)

        conditions_csv = _resolve_input_path(
            cfg.get("conditions_csv", "assets/conditions/gri30_small.csv"),
            config_parent=config_path.parent,
        )

        _preflight(
            baseline_mech=baseline_mech,
            candidate_mech=candidate_mech,
            phase=phase,
            conditions_csv=conditions_csv,
        )
        conditions = load_conditions(conditions_csv)

        mix = cfg.get("mixture") or {}
        fuel = str(mix.get("fuel", "CH4:1.0"))
        oxidizer = str(mix.get("oxidizer", "O2:1.0, N2:3.76"))

        integ = cfg.get("integration") or {}
        n_steps = int(integ.get("n_steps", 400))
        if n_steps < 2:
            raise ValueError("integration.n_steps must be >= 2")

        qoi = cfg.get("qoi") or {}
        species_last = list(
            qoi.get("species_last")
            or ["CO2", "CO", "H2O", "CH4", "O2", "NO", "NO2"]
        )
        species_max = list(qoi.get("species_max") or ["OH", "HO2"])

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
        )
        write_rows_csv(out_dir / "baseline_results.csv", baseline_rows)

        summary: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "ok",
            "failure_reason": None,
            "run_id": run_id,
            "config": str(config_path),
            "conditions_csv": str(conditions_csv),
            "phase": phase,
            "baseline_mechanism": str(baseline_mech),
            "baseline_counts": mechanism_counts(baseline_mech, phase),
            "qoi_metrics_count": 0,
            "pass_cases": None,
            "failed_cases": None,
            "worst_case": None,
        }

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
            )
            write_rows_csv(out_dir / "candidate_results.csv", candidate_rows)

            comparison_rows, comp_summary = compare_rows(
                baseline_rows,
                candidate_rows,
                rel_eps=rel_eps,
                rel_tolerance=rel_tolerance,
            )
            write_rows_csv(out_dir / "comparison_results.csv", comparison_rows)

            summary["candidate_mechanism"] = str(candidate_mech)
            summary["candidate_counts"] = mechanism_counts(candidate_mech, phase)
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
