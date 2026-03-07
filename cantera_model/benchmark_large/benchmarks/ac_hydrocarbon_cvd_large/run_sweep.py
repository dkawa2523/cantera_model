from __future__ import annotations

import sys
from pathlib import Path
import csv
import time
import os

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.run_batch_surface import BatchSurfaceConfig, run_batch_surface


INTERFACE = "ac_surf"
SAVE_RATES = os.environ.get("SAVE_RATES", "0").strip() == "1"


def main() -> int:
    mech_path = Path(__file__).with_name("ac_hydrocarbon_cvd_large.yaml")
    cond_path = Path(__file__).with_name("conditions.csv")
    out_root = ROOT / "outputs" / "ac_hydrocarbon_cvd_large"
    out_root.mkdir(parents=True, exist_ok=True)

    with cond_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        case_id = row["case_id"]
        T = float(row["T_K"])
        P_torr = float(row["P_Torr"])
        X = row["composition"]
        t_end = float(row.get("t_end_s", 0.5))
        n_steps = int(float(row.get("n_steps", 400)))
        area = float(row.get("area", 1.0))

        P_pa = P_torr * 133.32236842105263
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
        print(f"[ac_hydrocarbon_cvd_large] Running {run_id} ...")
        run_batch_surface(cfg, save_rates=SAVE_RATES)

    print(f"[ac_hydrocarbon_cvd_large] Done. Outputs: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
