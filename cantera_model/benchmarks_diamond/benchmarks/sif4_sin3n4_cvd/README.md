# Case: SiF4/NH3 -> Si3N4 CVD (surface + deposition interface)

This case is based on Cantera's official example for silicon nitride deposition.

- mechanism: `SiF4_NH3_mec.yaml`
- interface phase name: `SI3N4`
- typical conditions: T ~ 1600-1750 K, low pressure (~2 Torr)

Run a sweep:

```bash
python benchmarks/sif4_sin3n4_cvd/run_sweep.py
```


PFR (FlowReactor) driver (distance profiles):

```bash
python benchmarks/sif4_sin3n4_cvd/run_pfr.py
```
