# Cantera large semiconductor deposition benchmark add-on (SiF4/NH3 + large gas + multi-site surface)

This zip is meant to be **copied into** your `cantera_model` repository (or unzipped at the repo root).

It adds:

- `benchmarks/sif4_sin3n4_cvd_large/`  
  A larger benchmark derived from Cantera's SiF4/NH3 silicon nitride deposition example.

- `tools/build_sif4_large_mech.py`  
  YAML transformer that:
  - merges a large gas mechanism (default: `gri30.yaml`) into the base mechanism
  - clones the `SI3N4` surface mechanism across multiple site families

No external dependencies beyond what you already use (`python`, `pyyaml`, and `cantera` for running).

## Quick start

From your repo root:

```bash
python benchmarks/sif4_sin3n4_cvd_large/run_sweep.py
```

Mechanisms are downloaded and cached into `mechanisms/`.

## Verify mechanism size

After the first run, you can check the generated mechanism counts:

```bash
python tools/check_mechanism_size.py --mech mechanisms/SiF4_NH3_mec_large__gri30__multisite3__*.yaml --interface SI3N4
```

## What is physically "reasonable" here

- **All kinetic parameters (Arrhenius / sticking)** come from existing published mechanisms:
  - `SiF4_NH3_mec.yaml` (Cantera example deposition mechanism)
  - `gri30.yaml` (GRI-Mech 3.0)
- Multi-site surface expansion **does not invent new parameters**; it clones the same values across site families.
  - This is interpretable as multiple site types with identical kinetics (a common modeling approximation).

See `SOURCES.md` for provenance.
