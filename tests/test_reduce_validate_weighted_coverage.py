import numpy as np

from cantera_model.cli.reduce_validate import _compute_structural_balance_metrics


def test_compute_structural_balance_metrics_weighted_and_essential() -> None:
    nu_reduced = np.asarray(
        [
            [-1.0, 0.0],
            [1.0, -1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )
    weights = np.asarray([0.7, 0.2, 0.1, 0.0], dtype=float)
    essential = np.asarray([True, False, True, False], dtype=bool)
    metrics = _compute_structural_balance_metrics(
        nu_reduced=nu_reduced,
        species_after=4,
        reactions_after=2,
        activity_weights=weights,
        essential_cluster_mask=essential,
        balance_mode="hybrid",
    )
    assert abs(float(metrics["active_species_coverage"]) - 0.5) < 1.0e-12
    assert abs(float(metrics["weighted_active_species_coverage"]) - 0.9) < 1.0e-12
    assert abs(float(metrics["essential_species_coverage"]) - 0.875) < 1.0e-12
    assert str(metrics["balance_mode"]) == "hybrid"
