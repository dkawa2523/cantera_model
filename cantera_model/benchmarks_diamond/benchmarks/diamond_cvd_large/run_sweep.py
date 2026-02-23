from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import csv
import time
import os

from tools.fetch_mechanism import resolve_mechanism
from tools.run_batch_surface import BatchSurfaceConfig, run_batch_surface

# Local mechanism included in this pack
MECH_NAME = "diamond_gri30_multisite.yaml"
MECH_URLS: list[str] = []

# Ensure gri30.yaml is available for YAML import inside the mechanism
GRI_NAME = "gri30.yaml"
GRI_URLS = [
    "https://raw.githubusercontent.com/Cantera/cantera/main/data/gri30.yaml",
]

INTERFACE = "diamond_100_multi"
SAVE_RATES = os.environ.get("SAVE_RATES", "0").strip() == "1"


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    cache_dir = root / "mechanisms"

    # Make sure gri30.yaml is in cache_dir so the import works even if Cantera data path is missing it
    _ = resolve_mechanism(GRI_NAME, GRI_URLS, cache_dir)

    mech_path = resolve_mechanism(MECH_NAME, MECH_URLS, cache_dir, allow_cantera_lookup=False)

    cond_path = Path(__file__).with_name("conditions.csv")
    out_root = root / "outputs" / "diamond_cvd_large"
    out_root.mkdir(parents=True, exist_ok=True)

    with cond_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        case_id = row["case_id"]
        T = float(row["T_K"])
        P_atm = float(row["P_atm"])
        X = row["composition"]
        t_end = float(row.get("t_end_s", 0.1))
        n_steps = int(float(row.get("n_steps", 200)))
        area = float(row.get("area", 1.0))

        P_pa = P_atm * 101325.0

        run_id = f"{case_id}_{int(time.time())}"
        out_dir = out_root / run_id

        cfg = BatchSurfaceConfig(
            mech_path=str(mech_path),
            interface_name=INTERFACE,
            T=T,
            P_pa=P_pa,
            X=X,
            t_end=t_end,
            n_steps=n_steps,
            area=area,
            out_dir=str(out_dir),
        )
        print(f"[diamond_cvd_large] Running {run_id} ...")
        run_batch_surface(cfg, save_rates=SAVE_RATES)

    print(f"[diamond_cvd_large] Done. Outputs: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
