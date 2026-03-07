from cantera_model.reduction.merge_free import fragment_affinity


def test_fragment_affinity_examples() -> None:
    ch4 = {"name": "CH4", "composition": {"C": 1, "H": 4}}
    ch = {"name": "CH", "composition": {"C": 1, "H": 1}}
    n2 = {"name": "N2", "composition": {"N": 2}}

    cf4 = {"name": "CF4", "composition": {"C": 1, "F": 4}}
    f = {"name": "F", "composition": {"F": 1}}
    h2 = {"name": "H2", "composition": {"H": 2}}

    assert fragment_affinity(ch4, ch) > fragment_affinity(ch4, n2)
    assert fragment_affinity(cf4, f) > fragment_affinity(cf4, h2)
