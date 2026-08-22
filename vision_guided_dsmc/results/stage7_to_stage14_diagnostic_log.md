# Diagnostic research log: Stages 7–14

This log records executed follow-up experiments after the matched-cost Stage-6 benchmark. These are diagnostic results, not positive performance claims.

## Stage 7 — ten additional seeds at Kn = 0.20

Policy: original temperature/density/noise particle-allocation score.

- seeds: 101–110;
- improved cases: 7/10;
- mean adaptive/uniform error ratio: 1.01901;
- mean change: 1.90% worse;
- t-based 95% interval: [0.92895, 1.10907].

Conclusion: the three-seed improvement at Kn = 0.20 did not generalize.

## Stage 8 — ensemble reference and paired continuations

Method: three warm-state seeds, two independent high-particle reference runs per seed, and three paired adaptive/uniform continuation runs per warm state.

- paired comparisons: 9;
- improved pairs: 3/9;
- mean ratio: 1.09394;
- median ratio: 1.03554;
- mean change: 9.39% worse;
- t-based 95% interval: [0.95614, 1.23173].

Conclusion: reference noise alone does not explain the instability.

## Stage 9 — correlation of physical image features with local reference error

Nine coarse/reference comparisons over Kn = 0.05, 0.10, and 0.20.

Overall mean Spearman correlations:

- current combined gradient/noise score: 0.0466;
- temperature-gradient magnitude: 0.0513;
- density-gradient magnitude: -0.0086;
- raw temporal sigma(T): -0.0428;
- coefficient of variation sigma(T)/T: 0.1264;
- inverse particle count: -0.0009.

Conclusion: the current image score has almost no predictive relationship with local DSMC error.

## Stage 10 — disagreement between two independent coarse pilots

Overall mean Spearman correlations:

- combined two-run disagreement: 0.0154;
- temperature disagreement: 0.0164;
- speed disagreement: 0.0307;
- density disagreement: -0.0168;
- mean sigma(T)/T: 0.1533.

Conclusion: a second coarse run does not by itself produce a useful local error indicator.

## Stage 11 — batch-means uncertainty diagnostic

A reduced 6x6 diagnostic used four batch means to estimate the standard error of the sampled fields.

Overall mean Spearman correlations:

- combined batch standard-error score: 0.2312;
- relative batch SE(T): 0.2746;
- relative batch SE(rho): 0.0201;
- relative batch SE(speed): 0.1027.

Conclusion: batch SE(T)/T is more informative than gradients, but the correlation remains moderate.

## Stage 12 — matched-cost batch-SE particle allocation

Nine runs over the three Knudsen conditions.

- improved cases: 5/9;
- mean ratio: 1.01723;
- mean change: 1.72% worse;
- normal-approximation interval: [0.95466, 1.07980].

Conclusion: better feature/error correlation did not translate into robust closed-loop improvement.

## Stage 13 — batch-SE policy, ten seeds at Kn = 0.20

- improved cases: 3/10;
- mean ratio: 1.04166;
- median ratio: 1.05047;
- mean change: 4.17% worse;
- t-based 95% interval: [0.95245, 1.13086].

Conclusion: the batch-SE allocation also failed to generalize.

## Stage 14 — vision-guided collision-subcell refinement

Particle positions and weights were not reallocated. Instead, the collision subdivision map used 1x1, 2x2, or 3x3 subcells with the exact same global sum of subdivision-squared cost as the uniform 2x2 control.

- improved cases: 5/9;
- mean ratio: 1.05825;
- mean change: 5.82% worse;
- Kn = 0.05 mean ratio: 0.96535;
- Kn = 0.10 mean ratio: 1.03562;
- Kn = 0.20 mean ratio: 1.17377.

Conclusion: moving the vision decision from particle allocation to collision subcells did not provide robust benefit.

## Current scientific conclusion

The infrastructure, conservation tests, physical VHS/SBT cavity, matched-cost controls, and multi-seed execution are working. The present DSMC-only image features do not reliably predict where additional computational effort reduces error. Further manual tuning of noisy gradient features is not justified.

The next defensible route is to introduce a lower-noise external reference—DVM/Shakhov, a converged deterministic kinetic solution, or a substantially ensemble-averaged DSMC target—and train or calibrate the vision score against that reference. Any future method must outperform the equal-budget uniform control over multiple independent seeds.
