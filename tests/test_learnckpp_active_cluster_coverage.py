import numpy as np

from cantera_model.reduction.learnckpp.sparse_select import enforce_active_cluster_coverage


def test_enforce_active_cluster_coverage_with_swap_budget() -> None:
    nu = np.asarray(
        [
            [-1.0, -1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    importance = np.asarray([0.95, 0.9, 0.8, 0.7, 0.6], dtype=float)
    keep = np.asarray([True, False, False, False, False], dtype=bool)

    out, meta = enforce_active_cluster_coverage(
        keep=keep,
        importance=importance,
        nu_cand=nu,
        min_active_clusters=4,
        max_steps=8,
        max_keep_count=3,
    )
    assert int(np.sum(out)) <= 3
    assert int(meta["active_clusters"]) >= 4
    assert bool(meta["applied"]) is True
