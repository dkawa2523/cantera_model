import numpy as np

from cantera_model.reduction.learnckpp.sparse_select import select_sparse_overall


def test_select_sparse_overall_fallback_path() -> None:
    nu_cand = np.array(
        [
            [-1.0, 0.0, -1.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    t = np.linspace(0.0, 1.0, 20)
    rates = np.column_stack([0.5 + 0.1 * t, 0.2 + 0.05 * t, 0.03 + 0.01 * t])
    ydot_target = rates @ nu_cand.T
    features = np.column_stack([np.ones_like(t), t, t * t, np.sin(t), np.cos(t), np.ones_like(t)])

    keep, score = select_sparse_overall(
        nu_cand=nu_cand,
        ydot_target=ydot_target,
        features=features,
        cfg={
            "method": "hard_concrete",
            "target_keep_ratio": 0.3,
            "force_fallback": True,
            "fallback": {"select_method": "l1_threshold", "threshold_quantile": 0.75},
        },
    )

    assert keep.shape == (nu_cand.shape[1],)
    assert int(np.sum(keep)) >= 1
    assert score["fallback"] is True
    assert score["method_used"] == "l1_threshold"
