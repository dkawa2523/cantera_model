import numpy as np

from cantera_model.reduction.pooling.constraints import build_hard_mask, build_pairwise_cost
from cantera_model.reduction.pooling.graphs import build_species_graph
from cantera_model.reduction.pooling.train import train_pooling_assignment


def _species_meta() -> list[dict]:
    return [
        {"name": "CH4", "composition": {"C": 1, "H": 4}, "phase": "gas", "charge": 0, "radical": False, "role": "fuel"},
        {"name": "CH", "composition": {"C": 1, "H": 1}, "phase": "gas", "charge": 0, "radical": True, "role": "fuel"},
        {"name": "CF4", "composition": {"C": 1, "F": 4}, "phase": "gas", "charge": 0, "radical": False, "role": "etch"},
        {"name": "F", "composition": {"F": 1}, "phase": "gas", "charge": 0, "radical": True, "role": "etch"},
        {"name": "N", "composition": {"N": 1}, "phase": "gas", "charge": 0, "radical": True, "role": "carrier"},
        {"name": "H", "composition": {"H": 1}, "phase": "gas", "charge": 0, "radical": True, "role": "carrier"},
    ]


def test_pooling_training_returns_valid_assignment() -> None:
    meta = _species_meta()
    ns = len(meta)
    nr = 6
    nu = np.zeros((ns, nr), dtype=float)
    for j in range(nr):
        nu[j % ns, j] = -1.0
        nu[(j + 1) % ns, j] = 1.0

    F = np.zeros((ns, ns), dtype=float)
    F[0, 1] = 1.0
    F[2, 3] = 1.0
    graph = build_species_graph(nu, F, meta, {})

    rng = np.random.default_rng(8)
    features = rng.normal(size=(ns, 12))
    hard = build_hard_mask(meta, {"hard": {"element_overlap_required": True}})
    pair_cost = build_pairwise_cost(meta, {})
    out = train_pooling_assignment(
        graph,
        features,
        {"hard_mask": hard, "pair_cost": pair_cost, "species_meta": meta},
        {
            "model": {"backend": "numpy", "seed": 7},
            "train": {"target_ratio": 0.6, "temperature": 0.7},
            "constraints": {"hard_weight": 10.0, "soft_weight": 1.0},
        },
    )

    S = np.asarray(out["S"], dtype=float)
    assert S.shape[0] == ns
    assert np.allclose(np.sum(S, axis=1), 1.0)
    assert int(out["train_metrics"]["hard_ban_violations"]) == 0
