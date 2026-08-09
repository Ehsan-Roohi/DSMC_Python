# Gate 5 shock-triggered ignition screening bundle

This bundle searches for a condition in which a cold premixed hydrogen--oxygen
stream crosses an internal nozzle shock before it ignites.

Screening matrix:

- `p0 = 500 kPa`
- `T0 = 1000, 1250, 1500, 1750 K`
- `pb/p0 = 0.12, 0.18, 0.24`
- independent `Tback = 300 K`
- `2H2+O2+3Ar`
- chemistry ON
- 60,000 steps per case: 30,000 burn-in and 30,000 sampling
- 12 Slurm-array tasks, at most four concurrent

The submitter first runs a compile/physics preflight at the coldest,
highest-particle condition.  The screening array is released only when that
preflight passes.  A dependent summary job then creates:

- `QK_GATE5_SHOCK_IGNITION_SCREEN_REPORT.json`
- `QK_GATE5_SHOCK_IGNITION_RANKING.csv`
- one centerline CSV for every case
- the full flow, species, reaction, monitor and Tecplot outputs per case

The ranking prefers ignition 5--40 micrometres downstream of a detected shock
with upstream Mach at least 1.8.  Ignition requires `X_OH >= 1e-3`,
`X_H2O >= 1e-2`, and positive five-cell-coarse-grained heat release.
