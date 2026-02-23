from cantera_model.reduction.merge_free import build_candidate_mask


def test_element_overlap_hard_ban() -> None:
    species_meta = [
        {"name": "CH4", "composition": {"C": 1, "H": 4}, "phase": "gas", "radical": False},
        {"name": "CH", "composition": {"C": 1, "H": 1}, "phase": "gas", "radical": True},
        {"name": "N", "composition": {"N": 1}, "phase": "gas", "radical": True},
        {"name": "H", "composition": {"H": 1}, "phase": "surface", "radical": True},
    ]

    mask = build_candidate_mask(species_meta)
    assert bool(mask[0, 1]) is True  # CH4-CH
    assert bool(mask[2, 3]) is False  # N-H: no shared element
    assert bool(mask[1, 3]) is True  # CH-H shared H even across phase
