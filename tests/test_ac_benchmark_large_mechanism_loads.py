from pathlib import Path

import pytest


try:
    import cantera as ct
except Exception:  # pragma: no cover - environment dependent
    ct = None


def test_ac_benchmark_large_mechanism_loads() -> None:
    if ct is None:
        pytest.skip("cantera is required")

    mech = Path("cantera_model/benchmark_large/mechanisms/ac_hydrocarbon_cvd_large__gri30.yaml").resolve()
    iface = ct.Interface(str(mech), "ac_surf")
    gas = iface.adjacent["gas"]

    assert gas.n_species >= 50
    assert gas.n_reactions >= 100
    assert iface.n_species >= 20
    assert iface.n_reactions >= 100
    assert gas.n_species + iface.n_species >= 50
    assert gas.n_reactions + iface.n_reactions >= 100
