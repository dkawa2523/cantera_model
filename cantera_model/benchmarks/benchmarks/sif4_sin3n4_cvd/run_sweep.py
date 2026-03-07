from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import csv
from pathlib import Path
import time
import os

from tools.fetch_mechanism import resolve_mechanism
from tools.run_batch_surface import BatchSurfaceConfig, run_batch_surface


MECH_NAME = "SiF4_NH3_mec.yaml"
MECH_URLS = [
    # Cantera main repository (may work if internet is available)
    "https://raw.githubusercontent.com/Cantera/cantera/main/data/SiF4_NH3_mec.yaml",
    # A tag URL can be added here if you want a fixed version
    # "https://raw.githubusercontent.com/Cantera/cantera/v3.1.0/data/SiF4_NH3_mec.yaml",
]
INTERFACE = "SI3N4"
SAVE_RATES = os.environ.get("SAVE_RATES", "0").strip() == "1"


def main() -> int:
    root = Path(__file__).resolve().parents[2]  # pack root
    cache_dir = root / "mechanisms"
    mech_path = resolve_mechanism(MECH_NAME, MECH_URLS, cache_dir)

    cond_path = Path(__file__).with_name("conditions.csv")
    out_root = root / "outputs" / "sif4_sin3n4_cvd"
    out_root.mkdir(parents=True, exist_ok=True)

    with cond_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        case_id = row["case_id"]
        T = float(row["T_K"])
        P_torr = float(row["P_Torr"])
        X = row["composition"]
        t_end = float(row.get("t_end_s", 1.0))
        n_steps = int(float(row.get("n_steps", 200)))
        area = float(row.get("area", 1.0))

        # Torr -> Pa
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
        print(f"[sif4_sin3n4_cvd] Running {run_id} ...")
        run_batch_surface(cfg, save_rates=SAVE_RATES)

    print(f"[sif4_sin3n4_cvd] Done. Outputs: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
