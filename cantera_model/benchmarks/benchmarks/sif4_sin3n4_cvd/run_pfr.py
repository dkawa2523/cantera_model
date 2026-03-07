from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import csv
import json

from tools.fetch_mechanism import resolve_mechanism


MECH_NAME = "SiF4_NH3_mec.yaml"
MECH_URLS = [
    "https://raw.githubusercontent.com/Cantera/cantera/main/data/SiF4_NH3_mec.yaml",
]
INTERFACE = "SI3N4"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=1713.0, help="Temperature [K]")
    ap.add_argument("--P_torr", type=float, default=2.0, help="Pressure [Torr]")
    ap.add_argument("--X", type=str, default="SiF4:0.1427, NH3:0.8573", help="Inlet composition")
    ap.add_argument("--tube_d", type=float, default=5.08e-2, help="Tube diameter [m]")
    ap.add_argument("--u0", type=float, default=11.53, help="Inlet velocity [m/s]")
    ap.add_argument("--x_end", type=float, default=0.6, help="Reactor length [m]")
    ap.add_argument("--max_steps", type=int, default=5000, help="Safety cap on integration steps")
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    import numpy as np  # type: ignore
    import cantera as ct  # type: ignore

    root = Path(__file__).resolve().parents[2]
    mech_path = resolve_mechanism(MECH_NAME, MECH_URLS, root / "mechanisms")

    out_dir = Path(args.out_dir or (root / "outputs" / "sif4_sin3n4_cvd" / "pfr"))
    out_dir.mkdir(parents=True, exist_ok=True)

    P_pa = args.P_torr * 133.32236842105263

    iface = ct.Interface(str(mech_path), INTERFACE)
    gas = iface.adjacent["gas"]

    gas.TPX = args.T, P_pa, args.X
    iface.TP = args.T, P_pa

    # Geometry
    Ac = np.pi * args.tube_d**2 / 4.0

    # Flow reactor (1D PFR-like)
    fr = ct.FlowReactor(gas)
    fr.area = Ac
    fr.mass_flow_rate = gas.density * args.u0 * Ac
    fr.energy_enabled = False

    rsurf = ct.ReactorSurface(iface, fr)
    net = ct.ReactorNet([fr])

    gas_names = gas.species_names
    surf_names = iface.species_names

    # Deposition species (if present)
    dep_idx = {}
    for name in ["N(D)", "Si(D)"]:
        try:
            dep_idx[name] = iface.kinetics_species_index(name)
        except Exception:
            pass

    rows = []
    steps = 0
    while net.distance < args.x_end and steps < args.max_steps:
        net.step()
        steps += 1

        wdot = rsurf.kinetics.net_production_rates  # kmol/m^2/s for surface? depends on units
        dep = {k: float(wdot[idx]) for k, idx in dep_idx.items()}

        rows.append({
            "x": float(net.distance),
            "T": float(fr.T),
            "P": float(fr.thermo.P),
            "speed": float(fr.speed),
            **{f"Y_{sp}": float(y) for sp, y in zip(gas_names, gas.Y)},
            **{f"theta_{sp}": float(th) for sp, th in zip(surf_names, rsurf.coverages)},
            **{f"dep_{k}": float(v) for k, v in dep.items()},
        })

    # Save meta
    meta = {
        "mech_path": str(mech_path),
        "interface": INTERFACE,
        "T": args.T,
        "P_torr": args.P_torr,
        "X": args.X,
        "tube_d": args.tube_d,
        "u0": args.u0,
        "x_end": args.x_end,
        "steps": steps,
        "gas_n_species": gas.n_species,
        "gas_n_reactions": gas.n_reactions,
        "surf_n_species": iface.n_species,
        "surf_n_reactions": iface.n_reactions,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Save CSV
    if rows:
        cols = list(rows[0].keys())
        with (out_dir / "pfr_profile.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    print(f"[sif4_sin3n4_cvd:pfr] Done. steps={steps} out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
