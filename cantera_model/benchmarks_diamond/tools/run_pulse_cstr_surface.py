"""
ALD-like pulse/purge driver using a CSTR with surface chemistry.

This script provides a *generic* way to test:
- time-dependent inlet composition / flow (pulse A, purge, pulse B, purge...)
- surface reaction stiffness and coverage evolution
- suitability as a reduction benchmark (multiple operating "phases")

You can adapt the pulse schedule to your ALD chemistry (e.g., TMA/H2O) once you have the mechanism.

Implementation notes (Cantera Python API):
- Uses multiple inlet reservoirs with MassFlowControllers having time-dependent mdot(t).
- Reactor is IdealGasConstPressureReactor (isothermal, constant pressure).

Caveat:
- Exact API details may vary slightly across Cantera versions.
  The code targets Cantera 3.x series.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import csv
import json

import numpy as np  # type: ignore


@dataclass
class PulsePhase:
    name: str
    t0: float
    t1: float
    X: str
    mdot: float  # kg/s-equivalent in Cantera units (works as mass flow)


@dataclass
class PulseCSTRConfig:
    mech_path: str
    interface_name: str
    T: float
    P_pa: float
    reactor_volume: float
    area: float
    phases: List[PulsePhase]
    n_steps: int = 1000
    out_dir: str = "outputs/pulse_run"


def run_pulse_cstr(cfg: PulseCSTRConfig) -> None:
    import cantera as ct  # type: ignore

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    iface = ct.Interface(cfg.mech_path, cfg.interface_name)
    gas = iface.adjacent["gas"]

    # Initial state: use first phase composition
    gas.TPX = cfg.T, cfg.P_pa, cfg.phases[0].X

    # Reactor
    r = ct.IdealGasConstPressureReactor(gas, energy="off", volume=cfg.reactor_volume)
    rsurf = ct.ReactorSurface(iface, r, A=cfg.area)

    # Downstream reservoir
    env = ct.Reservoir(gas)

    # Create inlet reservoirs for each phase composition (deduplicate by X string)
    res_by_X: Dict[str, ct.Reservoir] = {}
    for ph in cfg.phases:
        if ph.X not in res_by_X:
            gtmp = ct.Solution(cfg.mech_path, gas.name)
            gtmp.TPX = cfg.T, cfg.P_pa, ph.X
            res_by_X[ph.X] = ct.Reservoir(gtmp)

    # Helper: mdot(t) for a given phase
    def make_mdot(ph: PulsePhase):
        def mdot(t):
            return ph.mdot if (ph.t0 <= t < ph.t1) else 0.0
        return mdot

    inlets = []
    for ph in cfg.phases:
        upstream = res_by_X[ph.X]
        mfc = ct.MassFlowController(upstream, r, mdot=make_mdot(ph))
        inlets.append(mfc)

    # Outlet controller (ties to first inlet as master)
    outlet = ct.PressureController(r, env, master=inlets[0], K=1e-5)

    sim = ct.ReactorNet([r])

    t_end = max(ph.t1 for ph in cfg.phases)
    times = np.linspace(0.0, t_end, int(cfg.n_steps))

    gas_names = gas.species_names
    surf_names = iface.species_names
    gas_X = np.zeros((len(times), len(gas_names)))
    surf_cov = np.zeros((len(times), len(surf_names)))

    for k, t in enumerate(times):
        sim.advance(t)
        gas_X[k, :] = gas.X
        surf_cov[k, :] = iface.coverages

    # Save meta + schedule
    meta = {
        "mech_path": cfg.mech_path,
        "interface_name": cfg.interface_name,
        "T": cfg.T,
        "P_pa": cfg.P_pa,
        "reactor_volume": cfg.reactor_volume,
        "area": cfg.area,
        "n_steps": cfg.n_steps,
        "phases": [ph.__dict__ for ph in cfg.phases],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Save time series
    with (out / "gas_X.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t"] + gas_names)
        for i, t in enumerate(times):
            w.writerow([float(t)] + [float(v) for v in gas_X[i, :]])

    with (out / "surface_coverages.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t"] + surf_names)
        for i, t in enumerate(times):
            w.writerow([float(t)] + [float(v) for v in surf_cov[i, :]])
