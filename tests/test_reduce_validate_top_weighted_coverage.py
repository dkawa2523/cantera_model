import numpy as np

from cantera_model.cli.reduce_validate import _compute_structural_balance_metrics, _evaluate_balance_gate


def test_top_weighted_coverage_metric_and_gate() -> None:
    nu_reduced = np.asarray(
        [
            [-1.0, 0.0],
            [1.0, -1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    S = np.eye(4, dtype=float)
    weights = np.asarray([0.7, 0.2, 0.1, 0.0], dtype=float)
    metrics = _compute_structural_balance_metrics(
        nu_reduced=nu_reduced,
        species_after=4,
        reactions_after=2,
        activity_weights=weights,
        essential_cluster_mask=np.asarray([True, False, False, False], dtype=bool),
        S_reduced=S,
        species_before=4,
        top_weight_mass_ratio=0.80,
        balance_mode="hybrid",
    )
    assert float(metrics["active_species_coverage_top_weighted"]) > float(metrics["weighted_active_species_coverage"])

    gate = _evaluate_balance_gate(
        metrics,
        {
            "enabled": True,
            "balance_mode": "hybrid",
            "min_reaction_species_ratio": 0.3,
            "max_reaction_species_ratio": 6.0,
            "min_active_species_coverage": 0.3,
            "min_weighted_active_species_coverage": 0.8,
            "min_active_species_coverage_top_weighted": 0.95,
            "min_essential_species_coverage": 0.6,
            "max_cluster_size_ratio": 0.8,
            "min_nu_rank_ratio": 0.3,
        },
    )
    assert gate["passed"] is True
