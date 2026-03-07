import numpy as np

from cantera_model.reduction.learnckpp.candidate_reactions import generate_overall_candidates


def _active_clusters(nu_cand: np.ndarray) -> int:
    if nu_cand.size == 0:
        return 0
    return int(np.sum(np.any(np.abs(nu_cand) > 1.0e-12, axis=1)))


def test_candidate_backfill_by_cluster_coverage_expands_active_clusters() -> None:
    species_meta = [{"name": f"S{i}", "composition": {"C": 1}} for i in range(6)]
    n = len(species_meta)
    nu = np.zeros((n, 1), dtype=float)
    S = np.eye(n, dtype=float)
    F = np.zeros((n, n), dtype=float)
    F[0, 1] = 9.0
    F[1, 0] = 8.0
    F[2, 3] = 1.0e-9
    F[3, 4] = 1.0e-9
    F[4, 5] = 1.0e-9

    base = generate_overall_candidates(
        nu=nu,
        F_bar=F,
        S=S,
        species_meta=species_meta,
        policy={
            "hard": {"element_overlap_required": True},
            "candidate": {
                "max_candidates": 8,
                "min_flux_quantile": 0.99,
                "min_candidates_floor": 2,
            },
        },
    )
    cov = generate_overall_candidates(
        nu=nu,
        F_bar=F,
        S=S,
        species_meta=species_meta,
        policy={
            "hard": {"element_overlap_required": True},
            "candidate": {
                "max_candidates": 8,
                "min_flux_quantile": 0.99,
                "min_candidates_floor": 2,
                "backfill_by_uncovered_clusters": True,
                "min_active_clusters": 4,
            },
        },
    )

    nu_base = np.asarray(base["nu_overall_candidates"], dtype=float)
    nu_cov = np.asarray(cov["nu_overall_candidates"], dtype=float)
    assert _active_clusters(nu_cov) >= 4
    assert _active_clusters(nu_cov) >= _active_clusters(nu_base)
