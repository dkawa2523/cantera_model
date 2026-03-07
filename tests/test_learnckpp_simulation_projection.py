import numpy as np

from cantera_model.reduction.conservation import conservation_violation
from cantera_model.reduction.learnckpp.rate_model import fit_rate_model
from cantera_model.reduction.learnckpp.simulate import simulate_reduced


def test_simulate_reduced_with_projection() -> None:
    time = np.linspace(0.0, 1.0, 50)
    features = np.column_stack([np.ones_like(time), time, time * time])

    rates_target = np.full((time.size, 1), 0.2, dtype=float)
    model = fit_rate_model(features, rates_target, {"model": "log_linear", "l2": 1.0e-6})
    model["sim_features"] = features.tolist()

    nu_sel = np.array([[-1.0], [1.0]], dtype=float)
    y0 = np.array([1.0, 0.0], dtype=float)
    A = np.array([[1.0, 1.0]], dtype=float)

    y, ydot = simulate_reduced(
        y0=y0,
        time=time,
        model_artifact=model,
        nu_overall_sel=nu_sel,
        proj_cfg={"enabled": True, "A": A, "reference": y0, "clip_nonnegative": True, "max_iter": 4},
    )

    assert y.shape == (time.size, 2)
    assert ydot.shape == (time.size, 2)
    assert float(np.min(y)) >= -1.0e-12
    assert conservation_violation(y, A, reference=y0) <= 1.0e-8
