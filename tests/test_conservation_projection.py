import numpy as np

from cantera_model.reduction.conservation import conservation_violation, project_to_conservation


def test_projection_reduces_violation() -> None:
    A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    X = np.array([[1.0, 0.2, 0.3], [0.3, 0.8, 0.1]])

    before = conservation_violation(X, A, reference=X[0])
    Xp = project_to_conservation(X, A, reference=X[0])
    after = conservation_violation(Xp, A, reference=Xp[0])

    assert after <= before + 1.0e-12
    assert after < 1.0e-8
