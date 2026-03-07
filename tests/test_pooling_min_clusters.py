import numpy as np

from cantera_model.reduction.pooling.constraints import build_hard_mask, build_pairwise_cost
from cantera_model.reduction.pooling.graphs import build_species_graph
from cantera_model.reduction.pooling.train import train_pooling_assignment


def test_pooling_respects_min_clusters() -> None:
    meta = [
        {"name": "A", "composition": {"C": 1}, "phase": "gas"},
        {"name": "B", "composition": {"C": 1}, "phase": "gas"},
        {"name": "C", "composition": {"C": 1}, "phase": "gas"},
        {"name": "D", "composition": {"C": 1}, "phase": "gas"},
        {"name": "E", "composition": {"C": 1}, "phase": "gas"},
        {"name": "F", "composition": {"C": 1}, "phase": "gas"},
    ]
    ns = len(meta)
    nu = np.zeros((ns, 6), dtype=float)
    for j in range(6):
        nu[j % ns, j] = -1.0
        nu[(j + 1) % ns, j] = 1.0
    F = np.zeros((ns, ns), dtype=float)
    graph = build_species_graph(nu, F, meta, {})
    # identical features tend to collapse to a small number of clusters before correction
    features = np.ones((ns, 8), dtype=float)
    hard = build_hard_mask(meta, {"hard": {"element_overlap_required": True}})
    cost = build_pairwise_cost(meta, {})
    out = train_pooling_assignment(
        graph,
        features,
        {"hard_mask": hard, "pair_cost": cost, "species_meta": meta},
        {
            "model": {"backend": "numpy", "seed": 3},
            "train": {"target_ratio": 0.3, "temperature": 0.7, "min_clusters": 4},
            "constraints": {"hard_weight": 10.0, "soft_weight": 1.0},
        },
    )
    assert int(out["train_metrics"]["n_clusters"]) >= 4
