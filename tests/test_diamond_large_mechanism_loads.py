from pathlib import Path

import pytest


try:
    import cantera as ct
except Exception:  # pragma: no cover - environment dependent
    ct = None


def test_diamond_large_mechanism_loads() -> None:
    if ct is None:
        pytest.skip("cantera is required")

    mech = Path("cantera_model/benchmarks_diamond/mechanisms/diamond_gri30_multisite.yaml").resolve()
    iface = ct.Interface(str(mech), "diamond_100_multi")
    gas = iface.adjacent["gas"]

    assert gas.n_species >= 50
    assert gas.n_reactions >= 100
    assert iface.n_species >= 10
    assert iface.n_reactions >= 50
    assert gas.n_species + iface.n_species >= 50
    assert gas.n_reactions + iface.n_reactions >= 100
