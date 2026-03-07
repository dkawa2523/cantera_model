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


def test_projection_pinv_precompute_is_numerically_stable() -> None:
    rng = np.random.default_rng(7)
    A = rng.normal(size=(4, 10))
    X = rng.normal(size=(50, 10))
    ref = X[0].copy()

    projected = project_to_conservation(X, A, reference=ref, clip_nonnegative=False, max_iter=3)

    gram = A @ A.T
    legacy = np.asarray(X, dtype=float).copy()
    target = A @ ref
    for t in range(legacy.shape[0]):
        x = legacy[t]
        for _ in range(3):
            resid = (A @ x) - target
            if float(np.max(np.abs(resid))) < 1.0e-12:
                break
            delta = np.linalg.pinv(gram) @ resid
            x = x - A.T @ delta
        legacy[t] = x

    assert np.max(np.abs(projected - legacy)) <= 1.0e-10
