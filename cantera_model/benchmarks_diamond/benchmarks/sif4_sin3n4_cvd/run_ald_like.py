from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.fetch_mechanism import resolve_mechanism
from tools.run_pulse_cstr_surface import PulseCSTRConfig, PulsePhase, run_pulse_cstr


MECH_NAME = "SiF4_NH3_mec.yaml"
MECH_URLS = [
    "https://raw.githubusercontent.com/Cantera/cantera/main/data/SiF4_NH3_mec.yaml",
]
INTERFACE = "SI3N4"


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    mech_path = resolve_mechanism(MECH_NAME, MECH_URLS, root / "mechanisms")

    # ALD-like schedule (illustrative only!)
    # pulse SiF4, purge, pulse NH3, purge
    phases = [
        PulsePhase("pulse_SiF4", 0.0, 0.05, "SiF4:1.0", mdot=1e-6),
        PulsePhase("purge_1",    0.05, 0.10, "NH3:1.0", mdot=0.0),
        PulsePhase("pulse_NH3",  0.10, 0.15, "NH3:1.0", mdot=1e-6),
        PulsePhase("purge_2",    0.15, 0.25, "NH3:1.0", mdot=0.0),
    ]

    cfg = PulseCSTRConfig(
        mech_path=str(mech_path),
        interface_name=INTERFACE,
        T=1713.0,
        P_pa=2.0 * 133.32236842105263,
        reactor_volume=1e-6,
        area=1.0,
        phases=phases,
        n_steps=800,
        out_dir=str(root / "outputs" / "sif4_sin3n4_cvd" / "ald_like"),
    )

    run_pulse_cstr(cfg)
    print("[sif4_sin3n4_cvd] ALD-like run done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
