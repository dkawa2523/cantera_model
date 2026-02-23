# Sources / references used in this benchmark

This benchmark is primarily intended for **mechanism reduction / reaction-network tooling**.
The gas mechanism is taken from Cantera's example-data; the surface mechanism is synthetic
but uses parameterizations and order-of-magnitude values that are consistent with common
surface-kinetics practice.

## Gas mechanism used

- Cantera example: n-hexane NUIG 2015 mechanism (`example_data/n-hexane-NUIG-2015.yaml`)
  - Cantera documentation mentions this as a detailed n-hexane mechanism with 1268 species.

## Surface kinetics parameterization

- Cantera documentation for *sticking reactions*:
  - Arrhenius-form sticking coefficient, relation to forward rate constant, and YAML syntax.
- Cantera documentation for reaction YAML syntax (spacing requirements, etc.).

## Order-of-magnitude values

- Preexponential factors for surface steps from transition-state / hard-sphere models:
  - Typical ranges for adsorption, diffusion, surface reactions, desorption are discussed.
- Example of hydrocarbon sticking coefficients on surfaces:
  - Reports typical sticking coefficients around ~0.02 for hydrocarbon molecules in a deposition context.

## Notes

- Surface species thermochemistry is set to zero (NASA7) because all surface reactions in this file
  are written as irreversible (=>). If you want reversible reactions and equilibrium consistency,
  you should replace the surface thermo with a physically consistent dataset.

