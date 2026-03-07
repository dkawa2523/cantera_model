# Mechanism sources

This pack fetches mechanisms from official Cantera repositories when they are not available locally.

- SiF4/NH3 Si3N4 CVD mechanism:
  - filename: `SiF4_NH3_mec.yaml`
  - upstream: Cantera main repository (`data/SiF4_NH3_mec.yaml`)
- Diamond CVD mechanism:
  - filename: `diamond.yaml`
  - upstream: Cantera main repository (`data/diamond.yaml`)
- Silicon-carbide example mechanism (optional):
  - filename: `silicon-carbide.yaml` (or `silicon_carbide.yaml` depending on Cantera version)
  - upstream: Cantera example-data repository

If you already have Cantera installed, these files often exist locally under Cantera's data directories.
