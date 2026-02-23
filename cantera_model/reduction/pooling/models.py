from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None
try:
    from torch_geometric.nn import GCNConv
except Exception:  # pragma: no cover
    GCNConv = None
try:  # pragma: no cover
    import tgp as _tgp  # noqa: F401
    TGP_AVAILABLE = True
except Exception:  # pragma: no cover
    TGP_AVAILABLE = False


class _NumpyPoolingModel:
    def __init__(self, *, n_clusters: int, seed: int = 0) -> None:
        self.n_clusters = int(max(1, n_clusters))
        self.seed = int(seed)


if nn is not None and torch is not None:
    class _TorchPoolingModel(nn.Module):  # type: ignore[misc]
        def __init__(self, input_dim: int, hidden_dim: int, n_clusters: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_clusters),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
else:  # pragma: no cover
    class _TorchPoolingModel:  # type: ignore[no-redef]
        pass


if nn is not None and torch is not None and GCNConv is not None:
    class _TorchGeometricPoolingModel(nn.Module):  # type: ignore[misc]
        def __init__(self, input_dim: int, hidden_dim: int, n_clusters: int, dropout: float) -> None:
            super().__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim, add_self_loops=True, normalize=True)
            self.conv2 = GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True)
            self.head = nn.Linear(hidden_dim, n_clusters)
            self.dropout = float(max(0.0, min(1.0, dropout)))

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None) -> torch.Tensor:
            h = self.conv1(x, edge_index, edge_weight)
            h = torch.relu(h)
            if self.dropout > 0.0:
                h = nn.functional.dropout(h, p=self.dropout, training=self.training)
            h = self.conv2(h, edge_index, edge_weight)
            h = torch.relu(h)
            return self.head(h)
else:  # pragma: no cover
    class _TorchGeometricPoolingModel:  # type: ignore[no-redef]
        pass


def build_pooling_model(input_dim: int, cfg: dict[str, Any] | None) -> Any:
    c = dict(cfg or {})
    model_cfg = dict(c.get("model") or {})
    n_clusters = int(model_cfg.get("n_clusters", 2))
    backend = str(model_cfg.get("backend", "pyg")).strip().lower()
    seed = int(model_cfg.get("seed", 0))
    hidden_dim = int(model_cfg.get("hidden_dim", max(16, input_dim)))
    dropout = float(model_cfg.get("dropout", 0.1))

    if backend in {"pyg", "torch_geometric", "tgp"} and torch is not None and nn is not None and GCNConv is not None:
        _ = TGP_AVAILABLE and backend == "tgp"
        return _TorchGeometricPoolingModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_clusters=n_clusters,
            dropout=dropout,
        )
    if backend in {"torch", "mlp"} and torch is not None and nn is not None:
        return _TorchPoolingModel(input_dim=input_dim, hidden_dim=hidden_dim, n_clusters=n_clusters)
    return _NumpyPoolingModel(n_clusters=n_clusters, seed=seed)


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(logits, dtype=float) / max(float(temperature), 1.0e-6)
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    den = np.sum(exp, axis=1, keepdims=True)
    den = np.where(den > 0.0, den, 1.0)
    return exp / den


def _edges_from_species_graph(graph: dict[str, Any], n_species: int) -> tuple[np.ndarray, np.ndarray]:
    edge_index = np.asarray(graph.get("edge_index"), dtype=np.int64)
    edge_weight = np.asarray(graph.get("edge_weight"), dtype=float)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
        edge_weight = np.ones((edge_index.shape[1],), dtype=float)
    if edge_index.shape[1] == 0:
        idx = np.arange(n_species, dtype=np.int64)
        edge_index = np.vstack([idx, idx])
        edge_weight = np.ones((n_species,), dtype=float)
    return edge_index, edge_weight


def _edges_from_bipartite_graph(graph: dict[str, Any], n_species: int) -> tuple[np.ndarray, np.ndarray]:
    sp = np.asarray(graph.get("species_index"), dtype=np.int64).reshape(-1)
    rx = np.asarray(graph.get("reaction_index"), dtype=np.int64).reshape(-1)
    ew = np.asarray(graph.get("edge_weight"), dtype=float).reshape(-1)
    if not (sp.shape == rx.shape == ew.shape):
        return _edges_from_species_graph({"edge_index": np.zeros((2, 0), dtype=np.int64)}, n_species)
    if sp.size == 0:
        return _edges_from_species_graph({"edge_index": np.zeros((2, 0), dtype=np.int64)}, n_species)

    by_rxn: dict[int, list[tuple[int, float]]] = {}
    for i in range(sp.size):
        sid = int(sp[i])
        rid = int(rx[i])
        if sid < 0 or sid >= n_species:
            continue
        by_rxn.setdefault(rid, []).append((sid, float(max(ew[i], 0.0))))

    edge_weights: dict[tuple[int, int], float] = {}
    for members in by_rxn.values():
        if not members:
            continue
        for i_pos, (si, wi) in enumerate(members):
            edge_weights[(si, si)] = edge_weights.get((si, si), 0.0) + wi
            for sj, wj in members[i_pos + 1 :]:
                w = wi * wj
                edge_weights[(si, sj)] = edge_weights.get((si, sj), 0.0) + w
                edge_weights[(sj, si)] = edge_weights.get((sj, si), 0.0) + w

    if not edge_weights:
        return _edges_from_species_graph({"edge_index": np.zeros((2, 0), dtype=np.int64)}, n_species)

    src: list[int] = []
    dst: list[int] = []
    val: list[float] = []
    for (i, j), w in edge_weights.items():
        src.append(int(i))
        dst.append(int(j))
        val.append(float(w))
    edge_index = np.asarray([src, dst], dtype=np.int64)
    edge_weight = np.asarray(val, dtype=float)
    mx = float(np.max(edge_weight)) if edge_weight.size else 0.0
    if mx > 0.0:
        edge_weight = edge_weight / mx
    return edge_index, edge_weight


def _graph_edges(graph: dict[str, Any], n_species: int) -> tuple[np.ndarray, np.ndarray]:
    g_type = str(graph.get("type", ""))
    if g_type == "species_graph":
        return _edges_from_species_graph(graph, n_species)
    if g_type == "bipartite_graph":
        return _edges_from_bipartite_graph(graph, n_species)
    return _edges_from_species_graph({"edge_index": np.zeros((2, 0), dtype=np.int64)}, n_species)


def infer_assignment(model: Any, graph: dict[str, Any], features: np.ndarray, cfg: dict[str, Any] | None) -> np.ndarray:
    feat = np.asarray(features, dtype=float)
    if feat.ndim != 2:
        raise ValueError("features must be 2-D")
    n_species = feat.shape[0]
    c = dict(cfg or {})
    train_cfg = dict(c.get("train") or {})
    temp = float(train_cfg.get("temperature", 0.7))

    if torch is not None and isinstance(model, _TorchPoolingModel):
        with torch.no_grad():
            x = torch.tensor(feat, dtype=torch.float32)
            logits = model(x).cpu().numpy()
        return _softmax(logits, temp)

    if torch is not None and isinstance(model, _TorchGeometricPoolingModel):
        edge_index, edge_weight = _graph_edges(graph, n_species)
        with torch.no_grad():
            x = torch.tensor(feat, dtype=torch.float32)
            ei = torch.tensor(edge_index, dtype=torch.long)
            ew = torch.tensor(edge_weight, dtype=torch.float32) if edge_weight.size else None
            logits = model(x, ei, ew).cpu().numpy()
        return _softmax(logits, temp)

    if isinstance(model, _NumpyPoolingModel):
        k = int(max(1, min(model.n_clusters, n_species)))
        rng = np.random.default_rng(model.seed)
        if n_species <= k:
            logits = np.eye(n_species, dtype=float)
            if n_species < k:
                logits = np.pad(logits, ((0, 0), (0, k - n_species)))
        else:
            # K-means like one-step assignment with deterministic/random centroids.
            idx = np.linspace(0, n_species - 1, num=k, dtype=int)
            centroids = feat[idx].copy()
            centroids += rng.normal(0.0, 1.0e-6, size=centroids.shape)
            dist = np.linalg.norm(feat[:, None, :] - centroids[None, :, :], axis=2)
            logits = -dist
        return _softmax(logits, temp)

    raise TypeError("unsupported pooling model type")
