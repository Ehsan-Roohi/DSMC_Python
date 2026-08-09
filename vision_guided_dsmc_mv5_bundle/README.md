# Mohammadzadeh MV5 confirmatory selector benchmark

MV5 is a preregistered confirmatory stage built after the MV4 diagnostic.  It
does **not** retune on the four MV4 held-out outcomes.  Instead it generates 16
new DSMC trajectories at four new `(Kn, U_lid)` combinations and evaluates a
predeclared target-free selector.

## Scientific changes from MV4

- Uses the joint two-dimensional convex hull in `log10(Kn)` and `U_lid/100`;
  this fixes the coordinate-wise rectangle false positive identified in MV4.
- Trains one bounded condition-aware residual U-Net on balanced development
  seeds from all four locked MV3 conditions.
- Allows bounded vision only for budget 1 inside the development hull.
- Otherwise uses a classical fallback selected only from development
  validation data, subject to a maximum per-condition degradation ratio of
  1.05 relative to Raw.
- Tests on four entirely new condition combinations: `(0.075,150)`,
  `(0.075,300)`, `(0.1,200)`, and `(0.1,400)`.
- Produces publication-scale physical `T` and `u` field figures as PNG and PDF.

Heat flux is excluded from inputs, outputs, gates, and claims.

## Unity run

Run from the existing `vision_guided_dsmc` project after the verified MV3 and
MV4 stages are present:

```bash
bash vision_guided_dsmc_mv5_bundle/install_and_submit_unity.sh "$PWD"
```

The submission creates:

- `moh_mv5_ref`: 16 confirmatory DSMC trajectories, at most four concurrent;
- `moh_mv5_model`: four budget tasks after the references pass;
- `moh_mv5_post`: aggregate, recursive verifier, figures, and portable bundle.

Job IDs and paths are written to `LAST_MOHAMMADZADEH_VISION_MV5_JOB.env`.
The default output is
`results/mohammadzadeh_2012/mv5_confirmatory_selector`.
