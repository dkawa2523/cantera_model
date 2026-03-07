from cantera_model.cli.reduce_validate import _resolve_physics_profile, _resolve_reduction_floors


def test_resolve_reduction_floors_profile_specific() -> None:
    cfg = {
        "trace_h5": "artifacts/traces/sif4_benchmark_large_trace.h5",
        "physics_floors": {
            "min_species_abs": 8,
            "min_species_ratio": 0.08,
            "min_reactions_abs": 10,
            "min_reactions_ratio": 0.03,
        },
    }
    profile = _resolve_physics_profile(cfg, trace_meta={"source": "trace_h5", "species_names": ["SIF4", "NH3"]})
    floors = _resolve_reduction_floors(profile, n_species=78, n_reactions=385)

    assert floors["profile"] == "benchmark_sif4_sin3n4_cvd"
    assert floors["min_species_after"] == 8
    assert floors["min_reactions_after"] == 12
