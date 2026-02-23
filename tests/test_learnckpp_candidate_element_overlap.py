import numpy as np

from cantera_model.reduction.learnckpp.candidate_reactions import generate_overall_candidates


def test_candidate_generation_enforces_zero_overlap_ban() -> None:
    species_meta = [
        {"name": "N", "composition": {"N": 1}},
        {"name": "H", "composition": {"H": 1}},
        {"name": "CH", "composition": {"C": 1, "H": 1}},
    ]
    n = len(species_meta)
    nu = np.zeros((n, 2), dtype=float)
    S = np.eye(n, dtype=float)
    F_bar = np.ones((n, n), dtype=float)
    np.fill_diagonal(F_bar, 0.0)

    out = generate_overall_candidates(
        nu=nu,
        F_bar=F_bar,
        S=S,
        species_meta=species_meta,
        policy={
            "hard": {"element_overlap_required": True},
            "candidate": {"max_candidates": 32, "min_flux_quantile": 0.0},
        },
    )
    meta = list(out["candidate_meta"])
    pairs = {(m["reactant_name"], m["product_name"]) for m in meta}

    assert ("N", "H") not in pairs
    assert ("H", "N") not in pairs
    assert ("H", "CH") in pairs or ("CH", "H") in pairs
