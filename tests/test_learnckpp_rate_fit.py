import numpy as np

from cantera_model.reduction.learnckpp.rate_model import fit_rate_model, predict_rates


def _signed_expm1(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.sign(arr) * np.expm1(np.abs(arr))


def test_fit_rate_model_log_linear() -> None:
    t = np.linspace(0.0, 1.0, 80)
    features = np.column_stack([np.ones_like(t), t, t * t])
    coef_true = np.array(
        [
            [0.20, -0.05],
            [0.15, 0.10],
            [-0.08, 0.06],
        ],
        dtype=float,
    )
    latent = features @ coef_true
    rates_target = _signed_expm1(latent)

    artifact = fit_rate_model(features, rates_target, {"model": "log_linear", "l2": 1.0e-10})
    pred = predict_rates(artifact, features)

    mse = float(np.mean((pred - rates_target) ** 2))
    assert mse < 1.0e-10
