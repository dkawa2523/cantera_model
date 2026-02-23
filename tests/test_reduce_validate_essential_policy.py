from cantera_model.cli.reduce_validate import _resolve_essential_species


def test_resolve_essential_species_prefers_explicit_when_qoi_keys_empty() -> None:
    qoi_cfg = {
        "species_last": ["A", "B"],
        "species_max": ["C"],
        "selectors": ["gas_X:D:final"],
    }
    species_meta = [
        {"name": "A"},
        {"name": "B"},
        {"name": "C"},
        {"name": "D"},
        {"name": "E"},
    ]
    cfg = {
        "balance_constraints": {
            "essential_qoi_keys": [],
            "include_selector_species_in_essential": False,
            "essential_species": ["E"],
            "max_essential_species": 1,
        }
    }
    out = _resolve_essential_species(qoi_cfg, species_meta, cfg)
    assert out == {"E"}
