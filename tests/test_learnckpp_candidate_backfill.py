import numpy as np

from cantera_model.reduction.learnckpp.candidate_reactions import generate_overall_candidates


def test_candidate_backfill_enforces_min_floor() -> None:
    species_meta = [
        {"name": "A", "composition": {"C": 1}},
        {"name": "B", "composition": {"C": 1}},
        {"name": "C", "composition": {"C": 1}},
        {"name": "D", "composition": {"C": 1}},
    ]
    n = len(species_meta)
    nu = np.zeros((n, 2), dtype=float)
    S = np.eye(n, dtype=float)
    F = np.zeros((n, n), dtype=float)
    F[0, 1] = 10.0
    F[1, 0] = 9.0
    F[2, 3] = 1.0e-9
    F[3, 2] = 1.0e-9

    out = generate_overall_candidates(
        nu=nu,
        F_bar=F,
        S=S,
        species_meta=species_meta,
        policy={
            "hard": {"element_overlap_required": True},
            "candidate": {
                "max_candidates": 64,
                "min_flux_quantile": 0.95,
                "min_candidates_floor": 4,
            },
        },
    )
    nu_cand = np.asarray(out["nu_overall_candidates"], dtype=float)
    assert nu_cand.shape[1] >= 4
