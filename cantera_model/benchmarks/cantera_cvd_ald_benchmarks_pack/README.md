# Cantera CVD/ALD Benchmark Pack (for Reduction / Model Compression)

This zip provides **ready-to-run benchmark drivers** for CVD-like surface chemistry systems
that are distributed with Cantera.

It is designed to be dropped into your `cantera_model` repository (or run standalone)
so you can:

- generate multi-case Cantera outputs (time series: X(t), coverages(t), optional ROP(t))
- compute network/stoichiometry matrices from these outputs in your pipeline
- benchmark reduction / merging / pruning methods

> Note: This pack is lightweight; mechanism YAML files are fetched on-demand.
> If you have Cantera installed, the mechanisms are typically already present.
> Otherwise, the scripts can download them from official sources.

---

## 0) Requirements

- Python 3.9+
- Cantera 3.0+ (pip or conda)
- numpy

---

## 1) Quick start (standalone)

From the pack root:

```bash
python benchmarks/sif4_sin3n4_cvd/run_sweep.py
python benchmarks/diamond_cvd/run_sweep.py
```

Outputs are written under `outputs/<case>/<run_id>/`.

---

## 2) How mechanism files are obtained

Each case uses `tools/fetch_mechanism.py`:

1) use local copy under `mechanisms/` if present  
2) else, try the installed Cantera data directories  
3) else, download from a list of URLs and cache into `mechanisms/`  

You can also set an override:

```bash
export CANTERA_MECH_PATH=/absolute/path/to/your/mechanism.yaml
```

---

## 3) Add to `cantera_model` repository

Recommended layout:

```
cantera_model/
  benchmarks/
    cantera_cvd_ald_benchmarks_pack/   # <-- unzip here
```

Then run from inside `cantera_model`:

```bash
python benchmarks/cantera_cvd_ald_benchmarks_pack/benchmarks/sif4_sin3n4_cvd/run_sweep.py
```

If your repo has a standard "cases/" folder, you can also move each folder under `benchmarks/`
into your existing structure. The scripts are self-contained.

---

## 4) Check size vs GRI30

Example:

```bash
python tools/check_mechanism_size.py --mech mechanisms/SiF4_NH3_mec.yaml --interface SI3N4
```

GRI30 reference (if available in your Cantera install):

```python
import cantera as ct
g = ct.Solution("gri30.yaml")
print(g.n_species, g.n_reactions)
```

---

## Licensing / attribution

Mechanism YAML files are part of the Cantera distribution and/or the official Cantera example-data
repository. When you use these mechanisms in publications, cite the original sources referenced
in each mechanism's `description` field and follow Cantera's guidance.

See: https://github.com/Cantera/cantera-example-data (Licensing and Attribution Notice)

## 1b) PFR-style benchmark (Si3N4 deposition)

Cantera provides an official plug-flow example for this mechanism; this pack includes a runnable PFR driver:

```bash
python benchmarks/sif4_sin3n4_cvd/run_pfr.py
```

This generates a `pfr_profile.csv` (distance profiles) suitable for reduction benchmarking.
