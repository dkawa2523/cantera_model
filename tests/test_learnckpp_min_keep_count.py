import numpy as np

from cantera_model.reduction.learnckpp.sparse_select import select_sparse_overall


def test_select_sparse_respects_min_keep_count() -> None:
    nu_cand = np.array(
        [
            [-1.0, 0.0, -1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0, -1.0, 1.0],
            [0.0, 1.0, 1.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    t = np.linspace(0.0, 1.0, 40)
    rates = np.column_stack([0.8 + 0.1 * t, 0.5 + 0.05 * t, 0.2 + 0.02 * t, 0.05 + 0.01 * t, 0.01 + 0.001 * t])
    ydot_target = rates @ nu_cand.T
    features = np.column_stack([np.ones_like(t), t, t * t, np.sin(t), np.cos(t), np.ones_like(t)])

    keep, score = select_sparse_overall(
        nu_cand=nu_cand,
        ydot_target=ydot_target,
        features=features,
        cfg={"method": "hard_concrete", "target_keep_ratio": 0.2, "lambda_l0": 1.0e-3, "min_keep_count": 3, "seed": 1},
    )
    assert int(np.sum(keep)) >= 3
    assert int(score["min_keep_count"]) == 3
