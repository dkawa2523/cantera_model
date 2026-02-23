"""
Generic isothermal batch reactor with a surface (CVD-style 0D benchmark driver).

- Uses IdealGasConstPressureReactor (pressure held constant)
- energy='off' (isothermal) for kinetics benchmarking
- Designed to output time series suitable for reaction-network/pathway analysis:
  - gas mole fractions (X)
  - surface coverages
  - (optional) net rates-of-progress (ROP) for gas and surface reactions

"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import numpy as np  # type: ignore


@dataclass
class BatchSurfaceConfig:
    mech_path: str
    interface_name: str
    T: float
    P_pa: float
    X: str  # Cantera composition string, e.g. "H2:0.95, C2H2:0.05"
    t_end: float = 0.5
    n_steps: int = 400
    area: float = 1.0  # surface area in cm^2 if mechanism uses cm
    out_dir: str = "outputs/run"


def run_batch_surface(cfg: BatchSurfaceConfig, save_rates: bool = False) -> None:
    import cantera as ct  # type: ignore

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    iface = ct.Interface(cfg.mech_path, cfg.interface_name)
    gas = iface.adjacent["gas"]

    gas.TPX = cfg.T, cfg.P_pa, cfg.X

    r = ct.IdealGasConstPressureReactor(gas, energy="off")
    _rsurf = ct.ReactorSurface(iface, r, A=cfg.area)
    sim = ct.ReactorNet([r])

    times = np.linspace(0.0, cfg.t_end, int(cfg.n_steps))
    gas_names = gas.species_names
    surf_names = iface.species_names

    gas_X = np.zeros((len(times), len(gas_names)))
    surf_cov = np.zeros((len(times), len(surf_names)))

    gas_rop = None
    surf_rop = None
    if save_rates:
        gas_rop = np.zeros((len(times), gas.n_reactions))
        surf_rop = np.zeros((len(times), iface.n_reactions))

    for k, t in enumerate(times):
        sim.advance(t)
        gas_X[k, :] = gas.X
        surf_cov[k, :] = iface.coverages
        if save_rates:
            gas_rop[k, :] = gas.net_rates_of_progress
            surf_rop[k, :] = iface.net_rates_of_progress

    meta = {
        "mech_path": cfg.mech_path,
        "interface_name": cfg.interface_name,
        "T": cfg.T,
        "P_pa": cfg.P_pa,
        "X": cfg.X,
        "t_end": cfg.t_end,
        "n_steps": cfg.n_steps,
        "area": cfg.area,
        "gas_phase": gas.name,
        "gas_n_species": gas.n_species,
        "gas_n_reactions": gas.n_reactions,
        "surf_phase": iface.name,
        "surf_n_species": iface.n_species,
        "surf_n_reactions": iface.n_reactions,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

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

    if save_rates and gas_rop is not None and surf_rop is not None:
        with (out / "gas_rop.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t"] + [f"R{i}" for i in range(gas.n_reactions)])
            for i, t in enumerate(times):
                w.writerow([float(t)] + [float(v) for v in gas_rop[i, :]])

        with (out / "surface_rop.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t"] + [f"SR{i}" for i in range(iface.n_reactions)])
            for i, t in enumerate(times):
                w.writerow([float(t)] + [float(v) for v in surf_rop[i, :]])
