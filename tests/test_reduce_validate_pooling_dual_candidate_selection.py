import numpy as np

import cantera_model.cli.reduce_validate as rv


def test_reduce_validate_pooling_dual_candidate_selection(monkeypatch, tmp_path) -> None:
    base_s = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    def _mock_train_pooling_assignment(graph, features, constraints, cfg):  # noqa: ANN001
        return {
            "S": base_s,
            "S_prob": base_s.copy(),
            "cluster_meta": [{"cluster_id": 0, "members": ["A", "B"], "elements": ["H"]}],
            "train_metrics": {
                "constraint_loss": 0.2,
                "hard_ban_violations": 0,
                "n_clusters": 2,
                "n_species": 4,
                "target_ratio": 0.5,
                "min_clusters": 2,
                "min_active_clusters": 2,
                "coverage_target": 0.4,
                "coverage_proxy": 0.4,
                "max_cluster_size_ratio": 0.75,
                "max_cluster_size_ratio_limit": 1.0,
                "cluster_guard_passed": True,
            },
            "model_info": {"model_type": "mock", "graph_type": "species_graph"},
        }

    def _mock_swap_refine(**kwargs):  # noqa: ANN003
        refined = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=float,
        )
        return {
            "improved": True,
            "S": refined,
            "steps_applied": 1,
            "coverage_proxy_before": 0.4,
            "coverage_proxy_after": 0.7,
            "constraint_loss_before": 0.2,
            "constraint_loss_after": 0.2,
            "max_cluster_size_ratio_after": 0.5,
        }

    monkeypatch.setattr(rv, "train_pooling_assignment", _mock_train_pooling_assignment)
    monkeypatch.setattr(rv, "_refine_cluster_balance_swap", _mock_swap_refine)

    species_meta = [
        {"name": "A", "composition": {"H": 1}, "phase": "gas"},
        {"name": "B", "composition": {"H": 1}, "phase": "gas"},
        {"name": "C", "composition": {"H": 1}, "phase": "gas"},
        {"name": "D", "composition": {"H": 1}, "phase": "gas"},
    ]
    mapping, metrics, saved = rv._fit_pooling_mapping(
        nu=np.zeros((4, 3), dtype=float),
        rop=np.zeros((5, 3), dtype=float),
        F_bar=np.eye(4, dtype=float),
        X=np.zeros((5, 4), dtype=float),
        wdot=np.zeros((5, 4), dtype=float),
        species_meta=species_meta,
        target_ratio=0.5,
        pooling_cfg={
            "graph": "species",
            "graph_cfg": {},
            "feature_cfg": {},
            "constraint_cfg": {"hard": {"element_overlap_required": True}},
            "model": {"backend": "pyg", "backend_candidates": ["pyg", "numpy"], "seed": 7},
            "candidate_selection": {
                "enabled": True,
                "pick_policy": "proxy_lexicographic",
                "dedupe_by_assignment_hash": True,
                "fallback_swap_refine": {"enabled": True, "max_steps": 8, "min_coverage_improve": 1.0e-6},
            },
            "train": {"max_cluster_size_ratio": 1.0},
        },
        artifact_path=tmp_path / "pooling.npz",
    )

    assert mapping.S.shape == (4, 2)
    assert saved is not None
    assert int(metrics["candidate_count"]) == 2
    assert int(metrics["candidate_unique_count"]) == 2
    assert str(metrics["candidate_selected_source"]) == "swap_refine"
    assert float(metrics["candidate_selected_coverage_proxy"]) > 0.4
    assert len(list(metrics.get("candidate_scores") or [])) == 2
