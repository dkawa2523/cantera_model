# Sources / provenance

This benchmark is constructed by reusing mechanisms distributed with Cantera.

## Base silicon nitride deposition mechanism

- File: `SiF4_NH3_mec.yaml`
- Source: Cantera example "Plug flow reactor: silicon nitride deposition" (gas + `SI3N4` interface)
  - https://cantera.org/examples/python/reactors/surf_pfr.py.html

## Large gas mechanism

- File: `gri30.yaml`
- Source: Cantera distribution of GRI-Mech 3.0
  - https://cantera.org/documentation/docs-3.0/sphinx/html/yaml/index.html (example input files list)

## Chemkin conversion

If you prefer to start from Chemkin format and convert to YAML:

- Cantera `ck2yaml` documentation
  - https://cantera.org/documentation/docs-3.0/sphinx/html/python/ck2yaml.html

> Note: the included benchmark build script works directly with YAML.
