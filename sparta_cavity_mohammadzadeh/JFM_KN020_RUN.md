# Matched SPARTA DSMC case for the JFM Kn=0.20 reviewer request

This workflow is intentionally separate from the `Kn=0.1` teaching and Ultra
campaigns. It generates the manuscript-matched high-Knudsen-number case and
submits eight independent Unity CPU jobs.

## Scientific contract

- `Kn = lambda_0/L = 0.20`, using the manuscript's equilibrium VHS mean free
  path at `T_w = 300 K` (the `gu_lambda_over_L` convention).
- Argon: `m = 6.6335e-26 kg`, `d_ref = 4.17e-10 m`, `T_ref = 273 K`,
  `omega = 0.81`, `alpha = 1`.
- Diffuse walls, full accommodation, `U_lid = 100 m/s`.
- `160 x 160` grid, 128 initial simulator particles/cell, 40,000 warmup steps,
  8,501 accumulated samples/cell, and eight independent seeds.
- Outputs include COM-subtracted temperature and COM-subtracted heat-flux
  density (`q_x`, `q_y`) for the DSMC/R13/R26 comparison.

The build dependency runs unit tests and a seconds-long SPARTA smoke case that
must exhibit exactly seven averaged fields before the ensemble is released.
Production members never overwrite an existing result directory.

## Unity launch

Use all eight seed slots for minimum wall-clock time:

```bash
DSMC_KN020_MAX_PARALLEL=8 bash <(curl -fsSL \
  https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/agent/sparta-kn020-jfm/sparta_cavity_mohammadzadeh/hpc/bootstrap_unity_sparta_kn020_jfm.sh)
```

The default is four simultaneous seeds. The bootstrap writes
`LAST_SPARTA_KN020_JFM_JOBS.env` at the campaign root. The collector always
creates a diagnostic archive, but succeeds only when all eight seeds and the
ensemble products are complete.

The article-ready comparison is a separate evidence step: use the raw ensemble
mean and 95% Student-t confidence intervals from the returned bundle in the
existing common-mask reviewer pipeline; do not style the quick-look plot as a
final paper figure without that comparison.
