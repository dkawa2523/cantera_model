from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cantera_model.eval.cantera_runner import load_conditions, run_case_trace
from cantera_model.io.trace_store import save_case_bundle
from cantera_model.network.stoich import build_nu, extract_species_meta
from cantera_model.types import CaseBundle, CaseTrace


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    return data


def _resolve_input_path(raw: Any, *, config_parent: Path) -> Path:
    path = Path(str(raw))
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (config_parent / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Cantera traces and save as HDF5 CaseBundle")
    parser.add_argument("--config", required=True, help="Path to trace config YAML")
    parser.add_argument("--run-id", default=None, help="Optional run id override")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)

    run_id = args.run_id or str(cfg.get("run_id") or f"trace_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")

    baseline_cfg = cfg.get("baseline") or {}
    mech_path = _resolve_input_path(
        baseline_cfg.get("mechanism", "assets/mechanisms/gri30.yaml"),
        config_parent=config_path.parent,
    )
    phase = str(baseline_cfg.get("phase", "gri30"))

    conditions_csv = _resolve_input_path(
        cfg.get("conditions_csv", "assets/conditions/gri30_tiny.csv"),
        config_parent=config_path.parent,
    )
    conditions = load_conditions(conditions_csv)

    mix = cfg.get("mixture") or {}
    fuel = str(mix.get("fuel", "CH4:1.0"))
    oxidizer = str(mix.get("oxidizer", "O2:1.0, N2:3.76"))

    integ = cfg.get("integration") or {}
    n_steps = int(integ.get("n_steps", 400))
    if n_steps < 2:
        raise ValueError("integration.n_steps must be >= 2")

    out_cfg = cfg.get("trace_output") or {}
    trace_root = _resolve_input_path(out_cfg.get("root", "artifacts/traces"), config_parent=config_path.parent)
    trace_path = trace_root / f"{run_id}.h5"

    nu, species_names, reaction_eqs = build_nu(mech_path, phase)
    species_meta = extract_species_meta(mech_path, phase=phase)

    cases: list[CaseTrace] = []
    for case in conditions:
        series = run_case_trace(
            mech_path,
            phase,
            case,
            fuel=fuel,
            oxidizer=oxidizer,
            n_steps=n_steps,
        )
        cases.append(
            CaseTrace(
                case_id=str(case["case_id"]),
                time=np.asarray(series["time"], dtype=float),
                temperature=np.asarray(series["temperature"], dtype=float),
                pressure=np.asarray(series["pressure"], dtype=float),
                X=np.asarray(series["X"], dtype=float),
                wdot=np.asarray(series["wdot"], dtype=float),
                rop=np.asarray(series["rop"], dtype=float),
                species_names=list(series["species_names"]),
                reaction_eqs=list(series["reaction_eqs"]),
                meta={"conditions": dict(case)},
            )
        )

    bundle = CaseBundle(
        mechanism_path=str(mech_path),
        phase=phase,
        species_names=species_names,
        reaction_eqs=reaction_eqs,
        cases=cases,
        meta={
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "conditions_csv": str(conditions_csv),
            "species_meta": species_meta,
            "nu": np.asarray(nu.toarray(), dtype=float).tolist(),
        },
    )
    save_case_bundle(trace_path, bundle)

    summary = {
        "status": "ok",
        "run_id": run_id,
        "trace_path": str(trace_path),
        "phase": phase,
        "mechanism": str(mech_path),
        "cases": len(cases),
        "species": len(species_names),
        "reactions": len(reaction_eqs),
    }
    trace_root.mkdir(parents=True, exist_ok=True)
    (trace_root / f"{run_id}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
