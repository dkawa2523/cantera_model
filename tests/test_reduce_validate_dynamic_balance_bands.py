import math

from cantera_model.cli.reduce_validate import _resolve_dynamic_balance_bands


def test_dynamic_balance_bands_diamond_profile_applies_bounded_relaxation() -> None:
    base = {
        "enabled": True,
        "profile": "benchmarks_diamond",
        "min_reaction_species_ratio": 0.40,
        "max_reaction_species_ratio": 4.00,
        "min_active_species_coverage": 0.50,
    }
    cfg = {
        "balance_constraints": {
            "dynamic": {
                "enabled": True,
                "profiles": {
                    "benchmarks_diamond": {
                        "complexity_offset_reactions": 100,
                        "complexity_span_reactions": 300,
                        "rules": {
                            "min_reaction_species_ratio": {"slope": -0.08, "floor": 0.30},
                            "max_reaction_species_ratio": {"slope": 2.00, "ceiling": 6.00},
                            "min_active_species_coverage": {"slope": -0.14, "floor": 0.38},
                        },
                    }
                },
            }
        }
    }

    out = _resolve_dynamic_balance_bands("benchmarks_diamond", base, species_before=78, reactions_before=385, cfg=cfg)

    # complexity=(385-100)/300 = 0.95
    assert out["balance_dynamic_applied"] is True
    assert math.isclose(float(out["balance_dynamic_complexity"]), 0.95, rel_tol=1e-9)
    assert math.isclose(float(out["min_reaction_species_ratio"]), 0.324, rel_tol=1e-9)
    assert math.isclose(float(out["max_reaction_species_ratio"]), 5.9, rel_tol=1e-9)
    assert math.isclose(float(out["min_active_species_coverage"]), 0.38, rel_tol=1e-9)


def test_dynamic_balance_bands_disabled_keeps_base_values() -> None:
    base = {
        "enabled": True,
        "profile": "benchmark_large",
        "min_reaction_species_ratio": 0.30,
        "max_reaction_species_ratio": 6.00,
        "min_active_species_coverage": 0.40,
    }
    cfg = {"balance_constraints": {"dynamic": {"enabled": False}}}

    out = _resolve_dynamic_balance_bands("benchmark_large", base, species_before=94, reactions_before=420, cfg=cfg)

    assert out["balance_dynamic_applied"] is False
    assert float(out["min_reaction_species_ratio"]) == 0.30
    assert float(out["max_reaction_species_ratio"]) == 6.00
    assert float(out["min_active_species_coverage"]) == 0.40
