from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from cantera_model.eval.conditions import load_conditions
from cantera_model.io.trace_store import save_case_bundle
from cantera_model.types import CaseBundle, CaseTrace

try:
    import cantera as ct
except Exception:  # pragma: no cover - runtime dependency
    ct = None


def _require_cantera() -> None:
    if ct is None:
        raise ImportError("cantera is required for run_surface_trace")


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
    # Allow built-in Cantera mechanism names like diamond.yaml
    if len(path.parts) == 1:
        return text
    return str(cfg_candidate)


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _surface_nu(iface: Any) -> np.ndarray:
    return np.asarray(iface.product_stoich_coeffs - iface.reactant_stoich_coeffs, dtype=float)


def _gas_nu_on_kinetics_axis(gas: Any, kinetics_species: list[str]) -> np.ndarray:
    gas_nu = np.asarray(gas.product_stoich_coeffs - gas.reactant_stoich_coeffs, dtype=float)
    out = np.zeros((len(kinetics_species), gas_nu.shape[1]), dtype=float)
    gas_idx = {name: i for i, name in enumerate(gas.species_names)}
    for k_idx, name in enumerate(kinetics_species):
        g_idx = gas_idx.get(name)
        if g_idx is not None:
            out[k_idx, :] = gas_nu[g_idx, :]
    return out


def _build_trace_reference(
    *,
    mechanism: str,
    interface_phase: str,
    gas_phase: str,
    include_gas_reactions: bool,
) -> tuple[list[str], list[str], np.ndarray, list[dict[str, Any]]]:
    iface = ct.Interface(mechanism, interface_phase)
    if gas_phase not in iface.adjacent:
        raise ValueError(f"gas phase not found in interface.adjacent: {gas_phase}")
    gas = iface.adjacent[gas_phase]

    species_names = list(iface.kinetics_species_names)
    nu_surface = _surface_nu(iface)
    surface_reaction_eqs = list(iface.reaction_equations())

    if include_gas_reactions:
        nu_gas = _gas_nu_on_kinetics_axis(gas, species_names)
        reaction_eqs = list(gas.reaction_equations()) + surface_reaction_eqs
        nu = np.concatenate([nu_gas, nu_surface], axis=1)
    else:
        reaction_eqs = surface_reaction_eqs
        nu = nu_surface

    if nu.shape != (len(species_names), len(reaction_eqs)):
        raise ValueError("nu shape mismatch in trace reference build")

    species_meta = _build_species_meta(iface)
    return species_names, reaction_eqs, nu, species_meta


def _build_species_meta(iface: Any) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    phase_objs = [iface] + [p for _, p in iface.adjacent.items()]
    for phase_obj in phase_objs:
        phase_name = str(getattr(phase_obj, "name", ""))
        for species in phase_obj.species():
            by_name[species.name] = {
                "name": species.name,
                "composition": {k: float(v) for k, v in dict(species.composition).items()},
                "phase": phase_name,
                "charge": float(getattr(species, "charge", 0.0) or 0.0),
                "radical": False,
                "role": "",
            }

    ordered = []
    for name in iface.kinetics_species_names:
        ordered.append(
            by_name.get(
                name,
                {
                    "name": name,
                    "composition": {},
                    "phase": "",
                    "charge": 0.0,
                    "radical": False,
                    "role": "",
                },
            )
        )
    return ordered


def _simulate_case(
    *,
    mechanism: str,
    interface_phase: str,
    gas_phase: str,
    case: dict[str, Any],
    n_steps_default: int,
    include_gas_reactions: bool,
    trace_wdot_policy: str,
    species_names_ref: list[str],
    reaction_eqs_ref: list[str],
    nu_reference: np.ndarray,
) -> CaseTrace:
    _require_cantera()
    iface = ct.Interface(mechanism, interface_phase)
    if gas_phase not in iface.adjacent:
        raise ValueError(f"gas phase not found in interface.adjacent: {gas_phase}")
    gas = iface.adjacent[gas_phase]

    gas.TPX = float(case["T_K"]), float(case["P_Pa"]), str(case["composition"])
    area = max(float(case.get("area", 1.0)), 1.0e-12)
    n_steps = max(int(case.get("n_steps", n_steps_default)), 2)
    t_end = max(float(case["t_end_s"]), 1.0e-12)

    reactor = ct.IdealGasConstPressureReactor(gas, energy="off")
    _ = ct.ReactorSurface(iface, reactor, A=area)
    net = ct.ReactorNet([reactor])

    kinetics_species = list(iface.kinetics_species_names)
    if kinetics_species != list(species_names_ref):
        raise ValueError("kinetics species names changed between probe and case simulation")

    kin_idx = {name: i for i, name in enumerate(kinetics_species)}
    gas_idx = {name: i for i, name in enumerate(gas.species_names)}
    surf_idx = {name: i for i, name in enumerate(iface.species_names)}

    other_phase_lookup: dict[str, tuple[Any, int]] = {}
    for _, phase_obj in iface.adjacent.items():
        idx_map = {name: i for i, name in enumerate(phase_obj.species_names)}
        for name in phase_obj.species_names:
            if name not in gas_idx and name not in surf_idx:
                other_phase_lookup[name] = (phase_obj, idx_map[name])

    def _state_vector() -> np.ndarray:
        vec = np.zeros((len(kinetics_species),), dtype=float)
        for name in kinetics_species:
            k = kin_idx[name]
            if name in surf_idx:
                vec[k] = float(max(0.0, iface.coverages[surf_idx[name]]))
            elif name in gas_idx:
                vec[k] = float(max(0.0, gas.X[gas_idx[name]]))
            elif name in other_phase_lookup:
                phase_obj, idx = other_phase_lookup[name]
                vec[k] = float(max(0.0, phase_obj.X[idx]))
        return vec

    nu_ref = np.asarray(nu_reference, dtype=float)
    if nu_ref.shape != (len(kinetics_species), len(reaction_eqs_ref)):
        raise ValueError("nu_reference shape mismatch with probe species/reactions")

    def _rate_vectors() -> tuple[np.ndarray, np.ndarray]:
        surface_rop = np.asarray(iface.net_rates_of_progress, dtype=float).copy()
        if include_gas_reactions:
            gas_rop = np.asarray(gas.net_rates_of_progress, dtype=float).copy()
            rop_vec = np.concatenate([gas_rop, surface_rop], axis=0)
        else:
            rop_vec = surface_rop

        if rop_vec.shape != (len(reaction_eqs_ref),):
            raise ValueError("rop vector length mismatch with reaction equations")

        if trace_wdot_policy == "stoich_consistent":
            wdot_vec = np.asarray(nu_ref @ rop_vec, dtype=float)
        elif trace_wdot_policy == "surface_only":
            wdot_vec = np.asarray(iface.net_production_rates, dtype=float).copy()
        else:
            raise ValueError(f"unsupported trace_wdot_policy: {trace_wdot_policy}")

        if wdot_vec.shape != (len(kinetics_species),):
            raise ValueError("wdot vector length mismatch with kinetics species")
        return wdot_vec, rop_vec

    times = [0.0]
    temps = [float(reactor.T)]
    press = [float(reactor.thermo.P)]
    x_rows = [_state_vector()]
    init_wdot, init_rop = _rate_vectors()
    wdot_rows = [init_wdot]
    rop_rows = [init_rop]

    for i in range(n_steps):
        target = t_end * float(i + 1) / float(n_steps)
        net.advance(target)
        times.append(float(net.time))
        temps.append(float(reactor.T))
        press.append(float(reactor.thermo.P))
        x_rows.append(_state_vector())
        wdot_vec, rop_vec = _rate_vectors()
        wdot_rows.append(wdot_vec)
        rop_rows.append(rop_vec)

    return CaseTrace(
        case_id=str(case["case_id"]),
        time=np.asarray(times, dtype=float),
        temperature=np.asarray(temps, dtype=float),
        pressure=np.asarray(press, dtype=float),
        X=np.asarray(x_rows, dtype=float),
        wdot=np.asarray(wdot_rows, dtype=float),
        rop=np.asarray(rop_rows, dtype=float),
        species_names=kinetics_species,
        reaction_eqs=list(reaction_eqs_ref),
        meta={"conditions": dict(case)},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate surface Cantera traces and save as HDF5 CaseBundle")
    parser.add_argument("--config", required=True, help="Path to surface trace config YAML")
    parser.add_argument("--run-id", default=None, help="Optional run id override")
    args = parser.parse_args()

    _require_cantera()

    config_path = Path(args.config).resolve()
    cfg = _load_yaml(config_path)
    run_id = args.run_id or str(cfg.get("run_id") or f"surface_trace_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")

    sim_cfg = dict(cfg.get("simulation") or {})
    if str(sim_cfg.get("mode", "surface_batch")) != "surface_batch":
        raise ValueError("simulation.mode must be 'surface_batch' for run_surface_trace")
    surface_cfg = dict(sim_cfg.get("surface") or {})
    interface_phase = str(surface_cfg.get("interface_phase", "diamond_100"))
    gas_phase = str(surface_cfg.get("gas_phase", "gas"))
    pressure_unit = str(surface_cfg.get("pressure_unit", "atm"))
    include_gas_reactions = _as_bool(surface_cfg.get("include_gas_reactions_in_trace"), default=False)
    trace_wdot_policy = str(surface_cfg.get("trace_wdot_policy", "surface_only")).strip().lower()
    if trace_wdot_policy not in {"surface_only", "stoich_consistent"}:
        raise ValueError("simulation.surface.trace_wdot_policy must be 'surface_only' or 'stoich_consistent'")

    baseline_cfg = dict(cfg.get("baseline") or {})
    mechanism = _resolve_mechanism_ref(
        surface_cfg.get("mechanism", baseline_cfg.get("mechanism", "diamond.yaml")),
        config_parent=config_path.parent,
    )

    conditions_csv = _resolve_input_path(
        cfg.get("conditions_csv", "cantera_model/benchmarks_diamond/benchmarks/diamond_cvd/conditions.csv"),
        config_parent=config_path.parent,
    )
    schema_cfg = {
        "pressure_unit": pressure_unit,
        "pressure_column": surface_cfg.get("pressure_column"),
        "temperature_column": surface_cfg.get("temperature_column"),
        "composition_column": surface_cfg.get("composition_column"),
        "time_column": surface_cfg.get("time_column"),
        "n_steps_column": surface_cfg.get("n_steps_column"),
        "area_column": surface_cfg.get("area_column"),
    }
    schema_cfg = {k: v for k, v in schema_cfg.items() if v not in (None, "")}
    conditions = load_conditions(conditions_csv, mode="surface_batch", schema_cfg=schema_cfg)

    integ = dict(cfg.get("integration") or {})
    n_steps_default = max(int(integ.get("n_steps", 120)), 2)

    out_cfg = dict(cfg.get("trace_output") or {})
    trace_root = _resolve_input_path(out_cfg.get("root", "artifacts/traces"), config_parent=config_path.parent)
    trace_root.mkdir(parents=True, exist_ok=True)
    trace_path = trace_root / f"{run_id}.h5"

    # Probe once to lock metadata and stoichiometric matrix for kinetics ordering.
    species_names, reaction_eqs, nu, species_meta = _build_trace_reference(
        mechanism=mechanism,
        interface_phase=interface_phase,
        gas_phase=gas_phase,
        include_gas_reactions=include_gas_reactions,
    )

    cases: list[CaseTrace] = []
    for case in conditions:
        cases.append(
            _simulate_case(
                mechanism=mechanism,
                interface_phase=interface_phase,
                gas_phase=gas_phase,
                case=case,
                n_steps_default=n_steps_default,
                include_gas_reactions=include_gas_reactions,
                trace_wdot_policy=trace_wdot_policy,
                species_names_ref=species_names,
                reaction_eqs_ref=reaction_eqs,
                nu_reference=nu,
            )
        )

    bundle = CaseBundle(
        mechanism_path=str(mechanism),
        phase=interface_phase,
        species_names=species_names,
        reaction_eqs=reaction_eqs,
        cases=cases,
        meta={
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "surface_trace_cli",
            "conditions_csv": str(conditions_csv),
            "interface_phase": interface_phase,
            "gas_phase": gas_phase,
            "include_gas_reactions_in_trace": include_gas_reactions,
            "trace_wdot_policy": trace_wdot_policy,
            "species_meta": species_meta,
            "nu": nu.tolist(),
        },
    )
    save_case_bundle(trace_path, bundle)

    summary = {
        "status": "ok",
        "run_id": run_id,
        "trace_path": str(trace_path),
        "mechanism": str(mechanism),
        "interface_phase": interface_phase,
        "gas_phase": gas_phase,
        "include_gas_reactions_in_trace": include_gas_reactions,
        "trace_wdot_policy": trace_wdot_policy,
        "cases": len(cases),
        "species": len(species_names),
        "reactions": len(reaction_eqs),
    }
    (trace_root / f"{run_id}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
