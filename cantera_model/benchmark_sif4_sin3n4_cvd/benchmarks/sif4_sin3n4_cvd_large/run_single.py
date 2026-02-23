from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import argparse
from pathlib import Path

from tools.fetch_mechanism import resolve_mechanism
from tools.run_batch_surface import BatchSurfaceConfig, run_batch_surface
from tools.build_sif4_large_mech import build_large_sif4_mech


BASE_MECH_NAME = "SiF4_NH3_mec.yaml"
BASE_MECH_URLS = [
    "https://raw.githubusercontent.com/Cantera/cantera/main/data/SiF4_NH3_mec.yaml",
]

EXTRA_GAS_MECH_NAME = "gri30.yaml"
EXTRA_GAS_MECH_URLS = [
    "https://raw.githubusercontent.com/Cantera/cantera/main/data/gri30.yaml",
]
INTERFACE = "SI3N4"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=1713.0)
    ap.add_argument("--P_torr", type=float, default=2.0)
    ap.add_argument("--X", type=str, default="SiF4:0.1427, NH3:0.8573")
    ap.add_argument("--t_end", type=float, default=1.0)
    ap.add_argument("--n_steps", type=int, default=300)
    ap.add_argument("--area", type=float, default=1.0)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--save-rates", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    cache_dir = root / "mechanisms"
    base_mech = resolve_mechanism(BASE_MECH_NAME, BASE_MECH_URLS, cache_dir)
    extra_mech = resolve_mechanism(EXTRA_GAS_MECH_NAME, EXTRA_GAS_MECH_URLS, cache_dir)
    mech_path = build_large_sif4_mech(
        base_mech_path=base_mech,
        extra_gas_mech_path=extra_mech,
        out_dir=cache_dir,
        interface_name=INTERFACE,
        site_families=("t", "s", "k"),
    )

    P_pa = args.P_torr * 133.32236842105263
    out_dir = args.out_dir or str(root / "outputs" / "sif4_sin3n4_cvd_large" / "single")

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
    print(f"[sif4_sin3n4_cvd_large] Done. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())