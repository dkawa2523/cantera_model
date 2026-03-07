from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_pooling_artifact(path: str | Path, artifact: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "cluster_meta": artifact.get("cluster_meta") or [],
        "train_metrics": artifact.get("train_metrics") or {},
        "model_info": artifact.get("model_info") or {},
    }
    np.savez_compressed(
        out,
        S=np.asarray(artifact.get("S"), dtype=float),
        S_prob=np.asarray(artifact.get("S_prob"), dtype=float),
        payload_json=json.dumps(payload, ensure_ascii=False),
    )
    return out


def load_pooling_artifact(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = np.load(p, allow_pickle=False)
    payload_raw = str(data["payload_json"].item())
    payload = dict(json.loads(payload_raw))

    return {
        "S": np.asarray(data["S"], dtype=float),
        "S_prob": np.asarray(data["S_prob"], dtype=float),
        "cluster_meta": list(payload.get("cluster_meta") or []),
        "train_metrics": dict(payload.get("train_metrics") or {}),
        "model_info": dict(payload.get("model_info") or {}),
    }
