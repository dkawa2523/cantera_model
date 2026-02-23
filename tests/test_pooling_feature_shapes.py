import numpy as np

from cantera_model.reduction.pooling.features import extract_reaction_features, extract_species_features


def _species_meta() -> list[dict]:
    return [
        {"name": "CH4", "composition": {"C": 1, "H": 4}, "phase": "gas", "charge": 0, "radical": False, "role": "fuel"},
        {"name": "CH", "composition": {"C": 1, "H": 1}, "phase": "gas", "charge": 0, "radical": True, "role": "fuel"},
        {"name": "CF4", "composition": {"C": 1, "F": 4}, "phase": "gas", "charge": 0, "radical": False, "role": "etch"},
        {"name": "F", "composition": {"F": 1}, "phase": "gas", "charge": 0, "radical": True, "role": "etch"},
    ]


def test_pooling_feature_shapes_contract() -> None:
    meta = _species_meta()
    ns = len(meta)
    nr = 5
    steps = 12

    X = np.abs(np.random.default_rng(3).normal(size=(steps, ns)))
    wdot = np.random.default_rng(4).normal(scale=1.0e-2, size=(steps, ns))
    phase_labels = [str(m.get("phase", "")) for m in meta]

    sp_feat = extract_species_features(meta, X, wdot, phase_labels, {})
    assert sp_feat.shape[0] == ns
    assert sp_feat.shape[1] >= 9

    nu = np.zeros((ns, nr), dtype=float)
    rop = np.abs(np.random.default_rng(5).normal(size=(steps, nr)))
    for j in range(nr):
        nu[j % ns, j] = -1.0
        nu[(j + 1) % ns, j] = 1.0

    rxn_feat = extract_reaction_features(nu, rop, {})
    assert rxn_feat.shape == (nr, 6)
