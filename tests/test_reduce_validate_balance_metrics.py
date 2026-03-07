import numpy as np

from cantera_model.cli.reduce_validate import _compute_structural_balance_metrics


def test_compute_structural_balance_metrics_basic() -> None:
    nu_reduced = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    metrics = _compute_structural_balance_metrics(nu_reduced=nu_reduced, species_after=4, reactions_after=3)
    assert abs(float(metrics["reaction_species_ratio"]) - 0.75) < 1.0e-12
    assert abs(float(metrics["active_species_coverage"]) - 0.75) < 1.0e-12
    assert float(metrics["nu_rank_ratio"]) > 0.60
