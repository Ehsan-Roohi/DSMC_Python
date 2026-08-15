# MV15C-A1 fresh `q_y` evaluation

## Reproducible artifact

- Stage: `MV15C_A1_Mohammadzadeh_qy_evaluation`
- Unity prediction job: `62948523` (`COMPLETED`, `0:0`)
- Unity post-processing job: `62948524` (`COMPLETED`, `0:0`)
- Archive: `MV15C_A1_QY_EVALUATION_BUNDLE_20260815T021623Z.zip`
- Archive SHA-256: `e1bbac11f65b758be80d574619908dc017c5b6add1376f4d4dbd8c73aa8995e9`
- Recursive artifact verification: all 23 tracked files verified

## Scientific decision

`MV15C_A1_fresh_q_y_supports_frozen_B3_DCIR_QY_with_original_temperature_QC_warning`

All ten pre-existing `q_y` gates passed. The predictor, B3 weights, seeds,
targets, baselines, metrics, and `q_y` gates were not changed after observing
the fresh trajectories. No DSMC trajectory was rerun or replaced.

The result is an outcome-blind, QC-amended fresh evaluation. It must not be
described as the unamended preregistered MV15C confirmation, because two of the
eight completed trajectories missed the original stochastic
temperature-extremum stationarity gate. Their mechanical, provenance,
checkpoint, and finite-field checks passed.

## Primary results

Ratios below are `q_y` NRMSE divided by the paired Raw B10 NRMSE; lower is
better and a ratio below 1 beats Raw B10.

| Method | `kn0p1_u400` | `kn0p08_u350` |
|---|---:|---:|
| Frozen selected B3 DCIR-QY | **0.9542** | **0.7506** |
| Raw B3 | 1.6560 | 1.6502 |
| Mamba B3 | 1.8861 | 1.1010 |
| DC-only B3 | 1.1594 | 0.8720 |
| TSVD B3 | 1.1223 | 0.9675 |
| Permuted-observation B3 | 3.2932 | 3.2751 |
| Raw B10 | 1.0000 | 1.0000 |

Relative to Raw B10, the selected B3 method reduced mean `q_y` NRMSE by
**4.58%** at the difficult extrapolation corner and **24.94%** at the new
near-corner condition. Relative to Mamba B3, the reductions were **49.41%**
and **31.82%**, respectively. Relative to Raw B3, they were **42.38%** and
**54.52%**.

## Per-seed result

Every fresh seed independently beat its paired Raw B10 comparator.

| Condition | Seed | Selected B3 / Raw B10 |
|---|---:|---:|
| `kn0p1_u400` | 151501 | 0.9596 |
| `kn0p1_u400` | 151502 | 0.9857 |
| `kn0p1_u400` | 151503 | 0.9199 |
| `kn0p1_u400` | 151504 | 0.9517 |
| `kn0p08_u350` | 151511 | 0.7489 |
| `kn0p08_u350` | 151512 | 0.7443 |
| `kn0p08_u350` | 151513 | 0.7380 |
| `kn0p08_u350` | 151514 | 0.7714 |

The two trajectories held by the original temperature QC were seed 151502
and seed 151513. Their `q_y` ratios were 0.9857 and 0.7380, so the QC warning
did not coincide with a poor `q_y` outcome.

For descriptive context only (not a replacement for the locked gates), the
mean of the four per-seed ratios and a small-sample t interval are 0.9542
[0.9111, 0.9973] at `kn0p1_u400`, and 0.7507 [0.7275, 0.7738] at
`kn0p08_u350`.

## Mechanistic interpretation

At `kn0p1_u400`, the uncorrected Mamba B3 field had a mean offset of
`1.0045e-2`; the selected data-consistent result reduced that to
`1.0754e-4`. Its fitted amplitude slope was 0.9815. The maximum absolute DC
error was `1.39e-17`, effectively machine precision. Thus the improvement is
consistent with the registered mechanism: retain the learned spatial
denoising while anchoring the observation-constrained mean/low-information
mode. It is not merely a smoother output or a metric-only artifact.

The field plots show the same behavior: the selected result preserves the
large-scale cavity structure, removes Raw B3 particle noise, and avoids an
obvious ringing, banding, or spatial-shift artifact.

## Claim boundary and recommended paper wording

Supported wording:

> In an outcome-blind QC-amended evaluation on eight fresh DSMC trajectories,
> the frozen B3 data-consistent estimator reduced mean wall-normal heat-flux
> NRMSE relative to paired Raw B10 by 4.6% at `Kn=0.1, U=400` and 24.9% at
> `Kn=0.08, U=350`; all eight seed-wise comparisons favored the estimator.

Do not write that the unamended MV15C preregistered confirmation passed. Its
reference-QC gate remains formally inconclusive because of two independent
temperature-extremum warnings. The valid conclusion is narrower and directly
about `q_y`: the frozen B3 estimator received fresh cross-seed support.

Using three sampling blocks instead of ten is a 70% reduction in the sampled
block budget (3.33x fewer blocks). It is not automatically a 70% reduction in
total DSMC wall time because burn-in and fixed simulation costs remain.
