from pathlib import Path

import pytest


try:
    import cantera as ct
except Exception:  # pragma: no cover - environment dependent
    ct = None


def test_sif4_large_mechanism_loads() -> None:
    if ct is None:
        pytest.skip("cantera is required")

    mech = Path(
        "cantera_model/benchmark_sif4_sin3n4_cvd/mechanisms/SiF4_NH3_mec_large__gri30__multisite3.yaml"
    ).resolve()
    iface = ct.Interface(str(mech), "SI3N4")
    gas = iface.adjacent["gas"]

    assert gas.n_species >= 50
    assert gas.n_reactions >= 100
    assert iface.n_species >= 10
    assert iface.n_reactions >= 10
    assert gas.n_species + iface.n_species >= 50
    assert gas.n_reactions + iface.n_reactions >= 100
