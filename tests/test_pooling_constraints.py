import numpy as np

from cantera_model.reduction.pooling.constraints import build_hard_mask, build_pairwise_cost, pooling_constraint_loss


def _species_meta() -> list[dict]:
    return [
        {"name": "CH4", "composition": {"C": 1, "H": 4}, "phase": "gas", "charge": 0, "radical": False, "role": "fuel"},
        {"name": "CH", "composition": {"C": 1, "H": 1}, "phase": "gas", "charge": 0, "radical": True, "role": "fuel"},
        {"name": "CF4", "composition": {"C": 1, "F": 4}, "phase": "gas", "charge": 0, "radical": False, "role": "etch"},
        {"name": "F", "composition": {"F": 1}, "phase": "gas", "charge": 0, "radical": True, "role": "etch"},
        {"name": "N", "composition": {"N": 1}, "phase": "gas", "charge": 0, "radical": True, "role": "carrier"},
        {"name": "H", "composition": {"H": 1}, "phase": "gas", "charge": 0, "radical": True, "role": "carrier"},
    ]


def test_pooling_constraints_hard_ban_and_loss() -> None:
    meta = _species_meta()
    hard = build_hard_mask(meta, {"hard": {"element_overlap_required": True}})

    idx = {m["name"]: i for i, m in enumerate(meta)}
    assert bool(hard[idx["CH4"], idx["CH"]])
    assert bool(hard[idx["CF4"], idx["F"]])
    assert not bool(hard[idx["N"], idx["H"]])

    pair_cost = build_pairwise_cost(meta, {})
    s_prob = np.eye(len(meta), dtype=float)
    loss = pooling_constraint_loss(s_prob, hard, pair_cost, {})
    assert float(loss) >= 0.0
