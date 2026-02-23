from cantera_model.reduction.pooling.constraints import build_hard_mask, build_surface_site_mask


def test_pooling_surface_site_family_constraint() -> None:
    meta = [
        {"name": "t_CH2(S)", "composition": {"C": 1, "H": 2}, "phase": "surface"},
        {"name": "s_CH2(S)", "composition": {"C": 1, "H": 2}, "phase": "surface"},
        {"name": "CH4", "composition": {"C": 1, "H": 4}, "phase": "gas"},
    ]
    cfg = {"hard": {"element_overlap_required": True, "phase_mixing_forbidden": True, "surface_site_family_strict": True}}
    site_mask = build_surface_site_mask(meta, cfg)
    hard = build_hard_mask(meta, cfg)

    assert bool(site_mask[0, 1]) is False
    assert bool(site_mask[0, 2]) is False
    assert bool(hard[0, 1]) is False
    assert bool(hard[0, 2]) is False
