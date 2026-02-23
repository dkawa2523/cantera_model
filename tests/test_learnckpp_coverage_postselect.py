import numpy as np

from cantera_model.reduction.learnckpp.sparse_select import coverage_aware_postselect


def test_coverage_aware_postselect_improves_weighted_coverage() -> None:
    nu_cand = np.asarray(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    keep = np.asarray([True, False, False, False], dtype=bool)
    importance = np.asarray([1.0, 0.9, 0.8, 0.7], dtype=float)
    out_keep, meta = coverage_aware_postselect(
        keep=keep,
        importance=importance,
        nu_cand=nu_cand,
        cfg={
            "enabled": True,
            "mode": "hybrid",
            "target_weighted_coverage": 0.8,
            "target_essential_coverage": 0.8,
            "cluster_weights": [0.4, 0.3, 0.2, 0.1],
            "essential_cluster_mask": [True, True, False, False],
            "max_keep_count": 3,
        },
    )
    assert int(np.sum(out_keep)) <= 3
    assert float(meta["weighted_coverage"]) >= 0.8
    assert float(meta["essential_coverage"]) >= 0.8
