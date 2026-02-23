from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import argparse
from pathlib import Path

from tools.fetch_mechanism import resolve_mechanism
from tools.run_batch_surface import BatchSurfaceConfig, run_batch_surface


MECH_NAME = "diamond.yaml"
MECH_URLS = [
    "https://raw.githubusercontent.com/Cantera/cantera/main/data/diamond.yaml",
]
INTERFACE = "diamond_100"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=1200.0)
    ap.add_argument("--P_atm", type=float, default=0.0263)
    ap.add_argument("--X", type=str, default="H:2e-03, H2:0.988, CH3:2e-04, CH4:0.01")
    ap.add_argument("--t_end", type=float, default=0.1)
    ap.add_argument("--n_steps", type=int, default=200)
    ap.add_argument("--area", type=float, default=1.0)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--save-rates", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    cache_dir = root / "mechanisms"
    mech_path = resolve_mechanism(MECH_NAME, MECH_URLS, cache_dir)

    P_pa = args.P_atm * 101325.0
    out_dir = args.out_dir or str(root / "outputs" / "diamond_cvd" / "single")

    cfg = BatchSurfaceConfig(
        mech_path=str(mech_path),
        interface_name=INTERFACE,
        T=args.T,
        P_pa=P_pa,
        X=args.X,
        t_end=args.t_end,
        n_steps=args.n_steps,
        area=args.area,
        out_dir=out_dir,
    )

    run_batch_surface(cfg, save_rates=args.save_rates)
    print(f"[diamond_cvd] Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())