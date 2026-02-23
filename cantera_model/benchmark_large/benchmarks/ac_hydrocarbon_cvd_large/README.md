# ac_hydrocarbon_cvd_large (gas >= 100 / rxn >= 500, surface >= 20 / rxn >= 100)

This benchmark is intended for **semiconductor deposition (CVD) / surface kinetics** workflows,
but with a **very large gas-phase mechanism** so that reaction-network reduction and
graph/ML-based pathway simplification can be stressed.

## What this case models (conceptually)

- **Gas phase:** uses Cantera's `example_data/n-hexane-NUIG-2015.yaml` mechanism (very large).
  - In Cantera's own example documentation, this mechanism is described as having **1268 species**.
- **Surface phase:** a synthetic microkinetic-like set of reactions representing carbon film growth
  from hydrocarbon fragments (CH4 / C2H2 / C2H4 / H2), expanded across **4 site families**
  to hit: **surface species >= 20** and **surface reactions >= 100**.

> Note: The surface scheme is **mass-balanced** and the rate parameters are chosen to be
> order-of-magnitude plausible, but they are **not fitted** to a specific material, reactor,
> or temperature window.

## Files

- `ac_hydrocarbon_cvd_large.yaml` : gas + bulk carbon + surface, using import of the n-hexane mechanism
- `conditions.csv` : example multi-condition sweep table
- `run_single.py` : run one 0D constant-pressure reactor with surface
- `run_sweep.py` : sweep conditions.csv

## Quickstart

```bash
python run_single.py
python run_sweep.py
```

If the gas mechanism cannot be found, verify that your Cantera installation includes example data
and that `ct.Solution("example_data/n-hexane-NUIG-2015.yaml")` works.

## Counts / validation

Run:

```bash
python ../../tools/check_mechanism_size.py ac_hydrocarbon_cvd_large.yaml
```

This will print:
- gas species / reactions
- surface species / reactions
- whether the interface can be constructed

## Design choices (reduction benchmark oriented)

- Surface species thermo is set to zeros (NASA7); surface reactions are **irreversible** to avoid
  dependence on surface equilibrium constants.
- Site families: T, S, K, D (e.g., terrace/step/kink/defect) are used to expand the surface network.
- Deposition is represented via production of `C_bulk` in the adjacent bulk phase.

