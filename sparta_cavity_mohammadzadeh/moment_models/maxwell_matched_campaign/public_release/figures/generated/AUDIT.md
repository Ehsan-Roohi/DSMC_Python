# DSMC sensitivity figure audit

## Scope

- Eight independent seeds per case, 8501 accumulated samples per cell.
- VHS argon, `omega=0.81`, `Kn=0.05`, `Uwall=100 m/s`, `Twall=300 K`.
- Baseline: `N160_ppc128`; all field differences are relative L2 norms on the common 160x160 target grid.
- Plot gates reproduce the predeclared campaign guides: 5% for low-order fields, 10% for heat flux/primary AF scalars, and 15% for fourth moments.

## Audited ranges

- T: 0.042--0.059%.
- velocity: 0.762--1.096%.
- stress: 2.701--3.841%.
- heat flux: 12.09--17.45%.
- R: 16.28--23.31%.
- Delta: 72.88--104.94%.
- Baseline scalars: f_AF_domain=0.04765625, mean_IAF_AF=0.32705513, PDelta_over_PR=0.04192090.

## Caveats

- The N120-to-N160 remapping uses linear boundary extrapolation at 636 target points because cell-centre extents differ.
- The plotted field errors use raw ensemble-mean fields. Heat flux and R are marginally outside their desired 10%/15% guides; Delta is sampling-limited and is not pointwise converged.
- The stable result is the qualitative tensorial dominance: PDelta/PR remains approximately 0.038--0.049. Exact AF support remains threshold- and sampling-sensitive.
- Panel (c) divides each scalar by its baseline value to place the three diagnostics on one non-misleading scale; absolute values are retained in `figure_data.csv`.
- A stale inherited `interpolation.caveat` string in the scalar JSON calls the fields legacy/single-realisation. It is contradicted by the same file's audited paths/schema and by the reducer outputs (eight independent realisations per case), so it is not propagated into the figure or caption.

## Input hashes

- `dsmc_sensitivity_vs_N160_ppc128.csv`: `ff970acef1633bd8e620907e6bc9903c71f046c66f9de56d8d5c7858e5dfcf70`
- `dsmc_sensitivity_vs_N160_ppc128.json`: `5ff4babc54a121e6ad6ddad9badb9623a09df962fc15218135e0c706b7aab13d`
- `dsmc_grid_particle_sensitivity.csv`: `6c41769c981597e2a1f2c8d1d2ee1483ecbfcb333194a5acc8230d5ffbc35192`
- `dsmc_grid_particle_sensitivity.json`: `2e27c7c494007ba68de936cc70ec87cc9f5d72383a98bc0f2049c0e8209d33ef`
