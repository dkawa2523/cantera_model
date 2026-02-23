import numpy as np

from cantera_model.network.flux import build_flux, reaction_importance


def test_flux_is_nonnegative() -> None:
    nu = np.array(
        [
            [-1.0, 0.0],
            [1.0, -1.0],
            [0.0, 1.0],
        ]
    )
    rop = np.array([[1.0, 0.5], [2.0, 1.5]])
    F = build_flux(nu, rop)
    assert F.shape == (3, 3)
    assert np.all(F >= 0.0)

    imp = reaction_importance(rop)
    assert imp.shape == (2,)
    assert np.all(imp > 0.0)
