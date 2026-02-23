from pathlib import Path

import numpy as np
import pytest

from cantera_model.network.stoich import build_element_matrix, build_nu, element_conservation_residual


pytest.importorskip("cantera")


def test_element_conservation_residual_small() -> None:
    mech = Path("assets/mechanisms/gri30.yaml")
    nu, species, _ = build_nu(mech, "gri30")
    A, _, _ = build_element_matrix(mech, species, phase="gri30")
    resid = element_conservation_residual(A, nu)
    assert np.max(np.abs(resid)) < 1.0e-8
