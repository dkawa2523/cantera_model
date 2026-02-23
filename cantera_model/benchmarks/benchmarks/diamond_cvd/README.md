# Case: Diamond film growth (CVD-like surface chemistry)

This case uses Cantera's `diamond.yaml`, which includes a small gas mechanism and a surface mechanism
for growth on diamond (100) surface.

- mechanism: `diamond.yaml`
- surface phase name: `diamond_100`
- adjacent phases: `gas`, `diamond` (bulk solid)

Run a sweep:

```bash
python benchmarks/diamond_cvd/run_sweep.py
```
