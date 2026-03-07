import numpy as np

from cantera_model.reduction.pooling.constraints import build_hard_mask, build_pairwise_cost
from cantera_model.reduction.pooling.graphs import build_species_graph
from cantera_model.reduction.pooling.train import train_pooling_assignment


def test_pooling_max_cluster_size_guard_repairs_large_cluster() -> None:
    meta = [{"name": f"S{i}", "composition": {"C": 1}, "phase": "gas"} for i in range(6)]
    ns = len(meta)
    nu = np.zeros((ns, 6), dtype=float)
    for j in range(6):
        nu[j % ns, j] = -1.0
        nu[(j + 1) % ns, j] = 1.0
    graph = build_species_graph(nu, np.zeros((ns, ns), dtype=float), meta, {})
    features = np.asarray(
        [
            [5.0, 0.0],
            [4.0, 0.0],
            [0.5, 0.0],
            [0.4, 0.0],
            [0.3, 0.0],
            [0.2, 0.0],
        ],
        dtype=float,
    )
    hard = build_hard_mask(meta, {"hard": {"element_overlap_required": True}})
    cost = build_pairwise_cost(meta, {})
    out = train_pooling_assignment(
        graph,
        features,
        {"hard_mask": hard, "pair_cost": cost, "species_meta": meta},
        {
            "model": {"backend": "numpy", "seed": 7},
            "train": {
                "target_ratio": 0.2,
                "temperature": 0.7,
                "min_clusters": 2,
                "min_active_clusters": 2,
                "coverage_target": 0.10,
                "coverage_max_clusters": 6,
                "max_cluster_size_ratio": 0.45,
            },
            "constraints": {"hard_weight": 10.0, "soft_weight": 1.0},
        },
    )
    tm = dict(out["train_metrics"])
    assert float(tm["max_cluster_size_ratio"]) <= 0.45 + 1.0e-12
    assert bool(tm["cluster_guard_passed"]) is True
    assert int(tm["n_clusters"]) >= 3
