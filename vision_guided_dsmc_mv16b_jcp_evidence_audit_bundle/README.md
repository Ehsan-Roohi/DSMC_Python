# MV16B JCP evidence audit

MV16B is an analysis-only continuation of the verified MV15C-A1 cavity result
and the MV16A engineering cylinder screen. It submits no DSMC trajectory and
does not train or tune a network.

The cavity audit adds frozen attribution arms, continuous Wiener controls,
two independent reference-half reconstructions, B-scaling/stationarity
diagnostics, top-wall and spectral errors, paired log-ratio intervals, exact
small-sample sign tests, and the explicitly assumption-dependent correction
for noise in a three-peer leave-one-out reference.

The cylinder audit applies the frozen DCIR modes only at native Bird/DS2V
fluid-cell centres. The least-squares geometry adapter is weighted by cell
area, contains no solid samples, and preserves the area-weighted DC exactly.
It reports both global `q_y` and near-wall normal kinetic heat flux `q_n`. The
latter is reconstructed from cell moments and is not mislabeled as a direct
wall-collision tally.

Every output preserves two limitations: the cylinder campaign stopped near
`tU/D=11.5` rather than 30, and four seeds cannot yield a one-sided exact sign
test below 0.0625. Consequently MV16B reports effect sizes and uncertainty;
it cannot manufacture a new preregistered confirmation.

The installer verifies dependencies, runs 12 deterministic unit tests, then
submits one CPU audit job after the existing MV16A prediction job and one
packaging job. A SHA256-verified ZIP and result pointer are returned to the
project root.

