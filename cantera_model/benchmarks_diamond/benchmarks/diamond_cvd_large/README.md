# Case: Diamond CVD (large gas + multi-site surface, reduction stress-test)

This benchmark is designed to satisfy:

- **Gas phase**: ≥ 40 species and ≥ 200 reactions
- **Surface phase**: ≥ 10 species and ≥ 50 reactions

## What it is

- Gas phase: **GRI-Mech 3.0** imported from `gri30.yaml`
  - uses Cantera YAML import syntax (`gri30.yaml/species: all`, `gri30.yaml/reactions: all`)
- Surface phase: based on Cantera's `diamond.yaml` surface micro-kinetics, but **replicated across 3 "site families"**
  to increase surface species / reactions while keeping a deposition-like pathway to solid carbon (`C(d)`).

> This is intended as a *benchmark for reduction algorithms* (pruning / merging / pooling / clustering),
> not as a validated semiconductor process model.

## Files

- Mechanism: `mechanisms/diamond_gri30_multisite.yaml`
- Surface/interface phase name: `diamond_100_multi`
- Adjacent phases: `gas`, `diamond`

## Run

Single run:

```bash
python benchmarks/diamond_cvd_large/run_single.py
```

Sweep (multiple conditions):

```bash
python benchmarks/diamond_cvd_large/run_sweep.py
```

To also save full rates-of-progress (large files):

```bash
SAVE_RATES=1 python benchmarks/diamond_cvd_large/run_sweep.py
```

## Quick size check

```bash
python tools/check_mechanism_size.py --mech mechanisms/diamond_gri30_multisite.yaml --gas gas --interface diamond_100_multi
```
