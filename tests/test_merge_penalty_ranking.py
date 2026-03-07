import numpy as np

from cantera_model.reduction.merge_free import build_similarity


def test_phase_charge_penalties_lower_score_not_hard_ban() -> None:
    species_meta = [
        {"name": "CH4", "composition": {"C": 1, "H": 4}, "phase": "gas", "charge": 0, "radical": False, "role": "fuel"},
        {"name": "CH", "composition": {"C": 1, "H": 1}, "phase": "gas", "charge": 0, "radical": True, "role": "fuel"},
        {"name": "H", "composition": {"H": 1}, "phase": "surface", "charge": 1, "radical": True, "role": "surface"},
    ]
    F = np.zeros((3, 3), dtype=float)
    sim = build_similarity(species_meta, F)

    # CH4-CH share elements with smaller penalty than CH4-H (phase/charge/role gap)
    assert sim[0, 1] > sim[0, 2]
    # still candidate (not hard-ban) because they share H
    assert sim[0, 2] > -1.0e8
