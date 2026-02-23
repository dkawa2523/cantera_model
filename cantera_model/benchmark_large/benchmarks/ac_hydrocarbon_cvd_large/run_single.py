from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse

from tools.run_batch_surface import BatchSurfaceConfig, run_batch_surface


INTERFACE = "ac_surf"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=1050.0)
    ap.add_argument("--P_torr", type=float, default=10.0)
    ap.add_argument("--X", type=str, default="H2:0.95, C2H2:0.05")
    ap.add_argument("--t_end", type=float, default=0.5)
    ap.add_argument("--n_steps", type=int, default=400)
    ap.add_argument("--area", type=float, default=1.0)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--save-rates", action="store_true")
    args = ap.parse_args()

    mech_path = Path(__file__).with_name("ac_hydrocarbon_cvd_large.yaml")
    P_pa = args.P_torr * 133.32236842105263

    out_dir = args.out_dir or str(ROOT / "outputs" / "ac_hydrocarbon_cvd_large" / "single")

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
    print(f"[ac_hydrocarbon_cvd_large] Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
