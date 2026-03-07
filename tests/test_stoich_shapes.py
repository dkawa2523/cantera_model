from pathlib import Path

import pytest

from cantera_model.network.stoich import build_nu


pytest.importorskip("cantera")


def test_nu_shape_matches_counts() -> None:
    mech = Path("assets/mechanisms/gri30.yaml")
    nu, species, reactions = build_nu(mech, "gri30")
    assert nu.shape == (len(species), len(reactions))
