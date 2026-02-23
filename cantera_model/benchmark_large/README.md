# ac_hydrocarbon_cvd_large_pack

This pack contains a **large gas+surface Cantera benchmark** for testing reaction-network analysis
and mechanism reduction (including *merge/aggregation* on surfaces).

## Benchmark case included

- `benchmarks/ac_hydrocarbon_cvd_large/`
  - `ac_hydrocarbon_cvd_large.yaml` (imports a very large gas mechanism from Cantera example_data)
  - `run_single.py`, `run_sweep.py`, `conditions.csv`

## Requirements

- Python + Cantera installed (v3.x or newer recommended).
- Cantera example data available (so that `example_data/n-hexane-NUIG-2015.yaml` can be loaded).

## Quick validation

```bash
python tools/check_mechanism_size.py --mech benchmarks/ac_hydrocarbon_cvd_large/ac_hydrocarbon_cvd_large.yaml --gas gas --interface ac_surf
```

## Outputs

Runs write CSV/JSON files into `outputs/` (created automatically).

