from cantera_model.reduction.merge_free import fit_merge_mapping


def test_stage_target_ratio_monotonic_species_reduction() -> None:
    species_meta = [
        {"name": "CH4", "composition": {"C": 1, "H": 4}},
        {"name": "CH", "composition": {"C": 1, "H": 1}},
        {"name": "C2H2", "composition": {"C": 2, "H": 2}},
        {"name": "CF4", "composition": {"C": 1, "F": 4}},
        {"name": "F", "composition": {"F": 1}},
    ]

    m_a = fit_merge_mapping(species_meta, None, target_ratio=0.8)
    m_b = fit_merge_mapping(species_meta, None, target_ratio=0.6)
    m_c = fit_merge_mapping(species_meta, None, target_ratio=0.4)

    assert m_a.S.shape[1] >= m_b.S.shape[1] >= m_c.S.shape[1]
