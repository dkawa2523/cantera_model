import numpy as np

from cantera_model.reduction.learnckpp.sparse_select import select_sparse_overall


def test_sparse_select_post_prune_refine_respects_constraints() -> None:
    nu_cand = np.array(
        [
            [-1.0, -1.0, 0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, -1.0, -1.0, 0.0, -1.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    t = np.linspace(0.0, 1.0, 40)
    rates = np.column_stack(
        [
            0.8 + 0.2 * t,
            0.5 + 0.1 * t,
            0.3 + 0.05 * t,
            0.2 + 0.02 * t,
            0.1 + 0.01 * t,
            0.05 + 0.01 * t,
        ]
    )
    ydot_target = rates @ nu_cand.T
    features = np.column_stack([np.ones_like(t), t, t * t, np.sin(t), np.cos(t), np.ones_like(t)])

    common_cfg = {
        "method": "l1_threshold",
        "target_keep_ratio": 0.95,
        "min_keep_count": 2,
        "min_active_clusters": 2,
        "coverage_aware_swap_steps": 4,
        "seed": 3,
        "coverage_postselect": {
            "enabled": True,
            "target_weighted_coverage": 0.60,
            "target_essential_coverage": 0.50,
            "cluster_weights": [1.0, 1.0, 1.0],
            "essential_cluster_mask": [True, False, True],
        },
    }

    cfg_no_refine = {
        **common_cfg,
        "post_prune_refine": {
            "enabled": False,
        },
    }
    keep_no, score_no = select_sparse_overall(
        nu_cand=nu_cand,
        ydot_target=ydot_target,
        features=features,
        cfg=cfg_no_refine,
    )

    cfg_refine = {
        **common_cfg,
        "post_prune_refine": {
            "enabled": True,
            "max_steps": 2,
            "importance_quantile_step": 0.40,
            "respect_coverage_postselect": True,
            "respect_active_cluster_coverage": True,
        },
    }
    keep_refine, score_refine = select_sparse_overall(
        nu_cand=nu_cand,
        ydot_target=ydot_target,
        features=features,
        cfg=cfg_refine,
    )

    assert int(np.sum(keep_refine)) <= int(np.sum(keep_no))
    assert bool(score_refine["post_prune_refine"]["enabled"]) is True
    assert int(score_refine["active_cluster_coverage"]["active_clusters_after_post_prune"]) >= int(
        score_refine["min_active_clusters"]
    )
    assert float(score_refine["weighted_active_species_coverage"]) + 1.0e-12 >= 0.60
    assert float(score_refine["essential_species_coverage"]) + 1.0e-12 >= 0.50
    assert keep_refine.shape == keep_no.shape
