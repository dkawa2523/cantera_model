import numpy as np

from cantera_model.reduction.learnckpp.candidate_reactions import generate_overall_candidates


def _species_meta() -> list[dict]:
    return [
        {"name": "CH4", "composition": {"C": 1, "H": 4}, "role": "fuel"},
        {"name": "CH", "composition": {"C": 1, "H": 1}, "role": "fuel"},
        {"name": "CF4", "composition": {"C": 1, "F": 4}, "role": "etch"},
        {"name": "F", "composition": {"F": 1}, "role": "etch"},
        {"name": "N", "composition": {"N": 1}, "role": "carrier"},
        {"name": "H", "composition": {"H": 1}, "role": "carrier"},
    ]


def test_generate_overall_candidates_includes_fragment_like_pairs() -> None:
    species_meta = _species_meta()
    n = len(species_meta)
    nu = np.zeros((n, 4), dtype=float)
    S = np.eye(n, dtype=float)
    F_bar = np.zeros((n, n), dtype=float)
    i_ch4 = 0
    i_ch = 1
    i_cf4 = 2
    i_f = 3
    F_bar[i_ch4, i_ch] = 10.0
    F_bar[i_ch, i_ch4] = 8.0
    F_bar[i_cf4, i_f] = 9.0
    F_bar[i_f, i_cf4] = 7.0

    out = generate_overall_candidates(
        nu=nu,
        F_bar=F_bar,
        S=S,
        species_meta=species_meta,
        policy={
            "hard": {"element_overlap_required": True},
            "candidate": {"max_candidates": 64, "min_flux_quantile": 0.0},
        },
    )

    nu_cand = np.asarray(out["nu_overall_candidates"], dtype=float)
    meta = list(out["candidate_meta"])
    assert nu_cand.shape[0] == n
    assert nu_cand.shape[1] == len(meta)
    assert nu_cand.shape[1] > 0

    pairs = {(m["reactant_name"], m["product_name"]) for m in meta}
    assert ("CH4", "CH") in pairs or ("CH", "CH4") in pairs
    assert ("CF4", "F") in pairs or ("F", "CF4") in pairs
