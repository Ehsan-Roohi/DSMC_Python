# Pure-Maxwell public analysis

`analyze_pure_maxwell.py` consumes the packaged common-grid primary fields,
recomputes lower-order and heat-flux errors, anti-Fourier topology/overlap and
the full smoothing/threshold sensitivity envelope, and regenerates six figure
pairs under `figures/pure_maxwell`.

The common-grid archive is the redistribution boundary needed because the
legacy R13 wall-completion solver cannot be republished without explicit
permission. The script does not import that solver and does not claim to
reconstruct the common-grid archive from raw R13 state alone. Independent
validators in `../validate_release.py` check the packaged raw DSMC fields,
accepted R13/R26 states, molecular-model locks and source hashes before this
reducer is run.

Outputs:

- `primary_k20`: density, temperature, velocity and heat-flux fields;
- `centerlines_k20`: six matched centreline profiles;
- `antifourier_k05` and `antifourier_k20`: common-support vector/topology
  diagnostics;
- `antifourier_atlas`: DSMC/R13/R26 cold-to-hot topology at both Knudsen
  numbers; and
- `model_metrics`: compact error and overlap summary.

