# Matched SPARTA DSMC case for the JFM Kn=0.20 reviewer request

This workflow is intentionally separate from the `Kn=0.1` teaching and Ultra
campaigns. It generates the manuscript-matched high-Knudsen-number case and
submits two independent, high-statistics Unity CPU jobs.

## Scientific contract

- `Kn = lambda_0/L = 0.20`, using the manuscript's equilibrium VHS mean free
  path at `T_w = 300 K` (the `gu_lambda_over_L` convention).
- Argon: `m = 6.6335e-26 kg`, `d_ref = 4.17e-10 m`, `T_ref = 273 K`,
  `omega = 0.81`, `alpha = 1`.
- Diffuse walls, full accommodation, `U_lid = 100 m/s`.
- `160 x 160` grid, 256 initial simulator particles/cell, 40,000 warmup steps,
  20,000 accumulated samples/cell, and two independent seeds (`104729` and
  `130363`).
- Outputs include COM-subtracted temperature and COM-subtracted heat-flux
  density (`q_x`, `q_y`) for the DSMC/R13/R26 comparison.

The build dependency runs unit tests and a seconds-long SPARTA smoke case that
must exhibit exactly seven averaged fields before the ensemble is released.
Production members never overwrite an existing result directory.

## Unity launch

Run both seed members concurrently for minimum wall-clock time:

```bash
DSMC_KN020_MAX_PARALLEL=2 bash <(curl -fsSL \
  https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/agent/sparta-kn020-jfm/sparta_cavity_mohammadzadeh/hpc/bootstrap_unity_sparta_kn020_jfm.sh)
```

The default is two simultaneous seeds. The bootstrap writes
`LAST_SPARTA_KN020_JFM_JOBS.env` at the campaign root. The collector always
creates a diagnostic archive, but succeeds only when both seeds and the
ensemble products are complete.

The article-ready comparison is a separate evidence step. With only two seeds,
the primary repeatability visualization is the seed-to-seed half-range; the
formal Student-t interval has only one degree of freedom and must not be sold as
a precise uncertainty estimate. Use the raw fields and repeatability products
in the existing common-mask reviewer pipeline.
