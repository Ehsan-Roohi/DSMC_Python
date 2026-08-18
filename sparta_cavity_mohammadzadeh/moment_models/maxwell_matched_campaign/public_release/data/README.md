# Released data

## Matched Maxwell cases

`dsmc/kn005` and `dsmc/kn020` contain the final cell-averaged SPARTA dump,
the complete input deck and its machine-readable case metadata. The dump
header lists `id`, `xc`, `yc` and exactly 15 `f_fieldavg` columns; consult
`case_metadata.json` for the ordered physical meaning and limitations of the
sampled moments.

`r13/kn005_N60` and `r13/kn020_N60` contain 17-field arrays in the state order
documented by `r13/equations/r13_maxwell_production.py`. The reports retain
the numerical acceptance record and explicitly do not claim external
validation.

`r26/kn005_N40` and `r26/kn020_N20` contain 17-field arrays plus coordinates,
Knudsen convention, lid speed and grid stretch in compressed NumPy archives.
Their run summaries embed the exact source manifest and nonlinear acceptance
gates.

## Reduced tables

- `pure_maxwell/common_grid_fields.npz`: six primary fields for DSMC, R13 and
  R26 on the common 160-by-160 grid at both matched Knudsen numbers;
- `pure_maxwell/field_metrics.csv`, `anti_fourier_metrics.csv`,
  `processing_sensitivity.csv` and `centerline_profiles.csv`: the complete
  numerical backing for the six matched-model figures;
- `dsmc_sensitivity_figure_data.csv`: compact plotted table for the earlier
  eight-realization VHS grid/PPC sensitivity study at `Kn = 0.05`;
- `dsmc_grid_particle_sensitivity.*` and
  `dsmc_sensitivity_vs_N160_ppc128.*`: audit-detail tables for that same
  sensitivity study, with portable source labels.

The sensitivity campaign is not relabelled as Maxwell-matched. The two
matched Maxwell SPARTA final fields are the directories above.
