# Results

Committed validation summaries live in this directory. Full particle-field
outputs are ignored because production runs are large. Each run writes:

- `fields.npz`
- `grid.csv`
- `lid_profile.csv`
- `history.csv`
- `metadata.json`

Never treat a quick-run output as a paper validation. The metadata records the
grid, particles per cell, time-step limits, seed, sample count, backend, and
collision probability diagnostics needed to distinguish them.

The committed small tables are:

- `collision_rate_verification.csv`: exact-all-pair Monte Carlo verification
  for the four non-TAS Bernoulli pair selectors;
- `cpu_grid_convergence.csv`: honest failed/passing CPU refinement evidence
  against the digitized Mohammadzadeh lid profiles;
- `cpu_kn01_ntc_50x50_lid_profile.csv`, `*_validation_metrics.json`, and
  `*_validation.png`: the raw profile, quantitative gate, and plot for the
  passing Kn=0.1 CPU baseline.
