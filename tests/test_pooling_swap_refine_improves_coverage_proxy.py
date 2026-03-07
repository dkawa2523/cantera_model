import numpy as np

from cantera_model.reduction.pooling.train import _refine_cluster_balance_swap


def test_pooling_swap_refine_improves_coverage_proxy() -> None:
    # Cluster 0 is heavy; moving one species to cluster 1 should improve coverage proxy.
    S = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    hard_mask = np.ones((4, 4), dtype=bool)
    pair_cost = np.zeros((4, 4), dtype=float)
    species_weights = np.array([3.0, 2.0, 1.0, 1.0], dtype=float)

    out = _refine_cluster_balance_swap(
        S=S,
        hard_mask=hard_mask,
        pair_cost=pair_cost,
        species_weights=species_weights,
        max_cluster_size_ratio=1.0,
        max_steps=16,
        min_coverage_improve=1.0e-9,
    )

    assert bool(out["improved"]) is True
    assert int(out["steps_applied"]) >= 1
    assert float(out["coverage_proxy_after"]) > float(out["coverage_proxy_before"])
    assert np.asarray(out["S"]).shape == S.shape
