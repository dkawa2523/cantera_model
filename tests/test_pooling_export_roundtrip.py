import numpy as np

from cantera_model.reduction.pooling.export import load_pooling_artifact, save_pooling_artifact


def test_pooling_export_roundtrip(tmp_path) -> None:
    path = tmp_path / "pooling_artifact.npz"
    src = {
        "S": np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=float),
        "S_prob": np.asarray([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]], dtype=float),
        "cluster_meta": [{"cluster_id": 0, "members": ["A", "C"]}, {"cluster_id": 1, "members": ["B"]}],
        "train_metrics": {"constraint_loss": 0.12, "hard_ban_violations": 0},
        "model_info": {"model_type": "_NumpyPoolingModel", "graph_type": "species_graph"},
    }

    out_path = save_pooling_artifact(path, src)
    loaded = load_pooling_artifact(out_path)

    assert np.allclose(loaded["S"], src["S"])
    assert np.allclose(loaded["S_prob"], src["S_prob"])
    assert loaded["cluster_meta"] == src["cluster_meta"]
    assert loaded["train_metrics"] == src["train_metrics"]
