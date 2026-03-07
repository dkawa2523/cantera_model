from cantera_model.reduction.merge_free import build_candidate_mask, build_phase_site_mask


def test_merge_free_phase_and_surface_site_mask() -> None:
    meta = [
        {"name": "CH4", "composition": {"C": 1, "H": 4}, "phase": "gas"},
        {"name": "CH", "composition": {"C": 1, "H": 1}, "phase": "gas"},
        {"name": "H_T(s)", "composition": {"H": 1}, "phase": "diamond_100"},
        {"name": "X_S(s)", "composition": {"H": 1}, "phase": "diamond_100"},
    ]
    policy = {"hard": {"phase_mixing_forbidden": True, "surface_site_family_strict": True}}
    phase_mask = build_phase_site_mask(meta, policy)
    cand = build_candidate_mask(meta, policy)

    assert bool(phase_mask[0, 1]) is True
    assert bool(phase_mask[0, 2]) is False
    assert bool(phase_mask[2, 3]) is False
    assert bool(cand[0, 1]) is True
    assert bool(cand[0, 2]) is False
