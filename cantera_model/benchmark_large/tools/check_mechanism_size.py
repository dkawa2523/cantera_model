"""
Print size information for a mechanism YAML (species / reactions).

Usage:
  python tools/check_mechanism_size.py --mech path/to/mech.yaml --gas gas --interface ac_surf

If --gas is omitted, this tool tries to load the default phase (first phase in the file).
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mech", required=True, help="Path to mechanism YAML")
    ap.add_argument("--gas", default=None, help="Gas phase name (optional)")
    ap.add_argument("--interface", default=None, help="Interface/surface phase name (optional)")
    args = ap.parse_args()

    import cantera as ct  # type: ignore

    mech = args.mech
    print(f"Mechanism: {mech}")

    if args.gas:
        gas = ct.Solution(mech, args.gas)
        print(f"[gas:{args.gas}] n_species={gas.n_species} n_reactions={gas.n_reactions}")

    if not args.gas:
        try:
            gas0 = ct.Solution(mech)
            print(f"[default phase:{gas0.name}] n_species={gas0.n_species} n_reactions={gas0.n_reactions}")
        except Exception as e:
            print(f"Could not load default phase: {e}", file=sys.stderr)

    if args.interface:
        iface = ct.Interface(mech, args.interface)
        print(f"[interface:{args.interface}] n_species={iface.n_species} n_reactions={iface.n_reactions}")
        try:
            for name, ph in iface.adjacent.items():
                print(f"[adjacent:{name}:{ph.name}] n_species={ph.n_species} n_reactions={ph.n_reactions}")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
