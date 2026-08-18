# JCP redesign: corrected gated run plan

## Decision

The manuscript can be redesigned for JCP, but the expensive DSMC campaign must
not start before a development-only gate.  The present cylinder result is
dominated by the 40-block prior, and the present references are too short to
support truth-level bias claims.  The new paper must therefore demonstrate
adaptive fusion under condition shift with converged, disjoint references.

## Corrections to the draft protocol

1. A fourth-order accumulator is enough to improve a direct variance estimate
   for temperature, but not for heat flux.  Since heat flux is third order, its
   particle-level variance generally requires raw moments and cross-moments up
   to sixth order.  Until those are implemented and verified, between-block
   scatter with B >= 2 is the primary heat-flux noise estimator; B=1 is a
   secondary extrapolation.
2. "Better in every unit" is not a valid universal requirement for a noisy
   adaptive estimator.  Primary decisions use a paired geometric-mean risk
   ratio and a predeclared confidence bound; the all-unit count is supporting
   evidence.
3. Consistency means absolute error approaches the raw estimator as B grows.
   It does not imply that the percentage gain over Raw-B grows monotonically.
4. Reference convergence must use effective block counts.  Four independent
   reference trajectories are preferred to 250 serial blocks from one seed.
5. Evaluation conditions are frozen only after the development pilot.  Existing
   conditions cannot be relabelled as fresh tests.

## Phase 0 — no new DSMC

- Recover the exact Unity source and configuration with `JCP0_collect.sh`.
- Implement the additive-moment noise ledger and verify the overlapping-block
  identities already visible in the archived data.
- Implement P-0, P-NN, P-NNs, P-NET, frozen gain, and blockwise EB gain in
  Cartesian DCT for the cavity and polar DCT for the cylinder.
- Use leave-one-condition-out development tests to choose three shift levels.
- Freeze bins, masks, priors, noise calibration, seeds, and scoring code before
  generating any new evaluation trajectory.

Go/no-go gate G0:

- raw/prior-only block ledger within 15% on average and no systematic trend;
- P-NET plus adaptive fusion risk ratio < 0.95 relative to both frozen gain and
  the better of prior-only and P-0 at the moderate shift;
- improvement in at least 75% of development units;
- no target-dependent tuning.

If G0 fails, stop the JCP redesign and use the current paper for a less demanding
venue after correcting attribution and reference-noise language.

The archived-array pilot already rejects P-NN+EB as the primary candidate: it
beats the prior but not the stronger P-0 envelope.  P-NET+EB is the promoted
candidate.  With 10x10 DCT bins it gives geometric error ratios to Raw-B10 of
0.604 and 0.630 at the two available cavity conditions, and it improves on the
prior/P-0 envelope in all 8 units.  Bin-width sensitivity from 8 to 25 modes per
axis preserves the same conclusion.  These values are development-only and
must not be quoted as prospective evidence.

## Phase 1 — cavity, only after G0

Three new conditions are selected by the frozen pilot: interior shift S1,
near-support-boundary shift S2, and outside-support shift S3.

For each condition:

- 8 evaluation seeds, each providing 14 disjoint late-time blocks:
  blocks 0-2 observation, block 3 guard, blocks 4-13 Raw-B10 comparator;
- 4 independent reference seeds, each providing at least 64 accepted blocks;
- all fields: n, u, v, T, Pxx, Pxy, Pyy, qx, qy;
- stationarity and block autocorrelation checked before opening references.

Primary cavity endpoint: deconvolved qy NRMSE at S2.  Temperature and velocity
are co-reported as the lower-order consistency check, not as a separate paper.

## Phase 2 — cylinder, only after cavity code passes dry-run

Mach is the preferred shift axis because the mesh and time-step logic can remain
closest to the verified Mach-10 configuration.

- M=8 development reference: 4 seeds x at least 40 accepted blocks.
- Existing M=10 data remain development data.
- M=12 prospective evaluation: 8 new evaluation seeds x 14 disjoint blocks.
- M=12 converged reference: 4 independent seeds x at least 64 accepted blocks.
- M=14 is optional and starts only if the M=12 gate passes.
- Extract native n, u, v, T, pressure tensor, qx, qy, qn, qt, and direct wall
  heat-flux tally from every accepted output.

Primary cylinder endpoints: global qy and near-wall qn at M=12.  The direct wall
tally is a validation endpoint and must not be substituted by a cell-centred
radial projection.

## Phase 3 — learning and paper

- Retrain the bounded restorer with mixed B in {1,2,3,5,10} and B/noise-level
  conditioning on development conditions only.
- Compare restorer-alone with restorer plus adaptive fusion.
- Freeze predictions before reference fields are unblinded.
- Main paper target length: about 35 pages; per-seed tables and extensive
  controls move to supplementary material.

## First action

Run `JCP0_collect.sh` on Unity and upload `JCP0_src.tar.gz` plus its SHA-256
file.  The next step is a source-specific Phase-0 patch and a single submission
command.  No M=8, M=12, or new cavity DSMC job should be submitted before that
patch passes its dry-run tests.
