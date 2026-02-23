# Case: Large SiF4/NH3 -> Si3N4 CVD benchmark (gas+surface)

This benchmark is built from Cantera's **SiF4/NH3 silicon nitride deposition** example mechanism and **augmented** to create a larger reduction stress-test while keeping physically sourced rate constants.

What this case does:

1) **Base mechanism**: `SiF4_NH3_mec.yaml` (Cantera example, gas + SI3N4 surface)
2) **Large gas augmentation**: merges `gri30.yaml` (GRI-Mech 3.0) into the gas phase
3) **Surface multi-site augmentation**: clones the `SI3N4` surface mechanism across multiple "site families" (`t_`, `s_`, `k_`) so that **surface species/reactions scale ~x3**

> Why this is useful:
> - Gas-phase: guarantees **species >= 40** and **reactions >= 200**.
> - Surface: guarantees **species >= 10** and **reactions >= 50**.
> - All kinetics come from published / widely used mechanisms; multi-site cloning preserves base Arrhenius / sticking parameters.

## How it works

The run scripts call `tools/build_sif4_large_mech.py` automatically:

- downloads / caches `SiF4_NH3_mec.yaml` and `gri30.yaml`
- generates `mechanisms/SiF4_NH3_mec_large__gri30__multisite3.yaml`
- runs the simulation against the generated mechanism

## Run a sweep

```bash
python benchmarks/sif4_sin3n4_cvd_large/run_sweep.py
```

To also save rates-of-progress CSVs:

```bash
SAVE_RATES=1 python benchmarks/sif4_sin3n4_cvd_large/run_sweep.py
```

## Run single condition

```bash
python benchmarks/sif4_sin3n4_cvd_large/run_single.py --T 1713 --P_torr 2 --X "SiF4:0.10, NH3:0.60, H2:0.20, CH4:0.05, O2:0.05"
```

## Run ALD-like pulse schedule

```bash
python benchmarks/sif4_sin3n4_cvd_large/run_ald_like.py
```

## Notes

- This is **not** intended as a fully validated semiconductor process model.
- The goal is to provide a **large, structured gas+surface benchmark** with a real deposition interface for:
  - network analysis
  - pruning/merging reduction
  - ML / GNN pooling experiments

