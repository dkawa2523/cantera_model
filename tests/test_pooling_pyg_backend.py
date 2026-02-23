import numpy as np
import pytest

from cantera_model.reduction.pooling.graphs import build_species_graph
from cantera_model.reduction.pooling.models import build_pooling_model, infer_assignment


def test_pooling_pyg_backend_assignment_shape() -> None:
    try:
        import torch  # noqa: F401
        from torch_geometric.nn import GCNConv  # noqa: F401
    except Exception:
        pytest.skip("torch-geometric unavailable")

    n_species = 6
    n_reactions = 8
    nu = np.zeros((n_species, n_reactions), dtype=float)
    for j in range(n_reactions):
        nu[j % n_species, j] = -1.0
        nu[(j + 1) % n_species, j] = 1.0

    graph = build_species_graph(
        nu,
        np.eye(n_species, dtype=float),
        [{"name": f"S{i}", "composition": {"H": 1}} for i in range(n_species)],
        {},
    )
    feat = np.random.default_rng(9).normal(size=(n_species, 10))

    model = build_pooling_model(
        input_dim=feat.shape[1],
        cfg={
            "model": {"backend": "pyg", "n_clusters": 3, "hidden_dim": 12, "seed": 7},
            "train": {"temperature": 0.7},
        },
    )
    S_prob = infer_assignment(model, graph, feat, {"train": {"temperature": 0.7}})

    assert S_prob.shape == (n_species, 3)
    assert np.allclose(np.sum(S_prob, axis=1), 1.0, atol=1.0e-6)
