# MV12 SAGE-QY safety-aware ensemble

MV10 made the primary `q_y` result worse than Raw DSMC `B=10`, so MV12 does
not spend another DSMC campaign and does not promote a larger neural network.
It reuses the completed, verified MV9/MV10 artifacts and asks a narrower
question: can development-only validation safely choose among the estimates we
already paid for?

SAGE-QY (Safety-Aware Gated Ensemble for `q_y`) uses six experts: Raw `B=1`,
Gaussian `B=1`, TSVD/POD `B=1`, MV9 NAFNet, MV9 Mamba, and MV10 multiscale.
For each physical condition it:

- chooses a single-expert anchor using development validation only;
- selects ridge strength by leave-one-block-out validation;
- fits deterministic nonnegative weights that sum exactly to one;
- falls back to the anchor unless the blend improves validation by at least
  0.5%; and
- measures target-free expert disagreement, abstaining to the anchor on
  high-disagreement samples.

Only `q_y/q_ref` is replaced. The other three MV9 Mamba channels are preserved
bitwise. The prediction job does not index the legacy evaluation targets and
hash-locks its output before a dependency-separated post job evaluates it.

## Scientific status

MV12 was designed after seeing MV9 and MV10 outcomes. Therefore its evaluation
on the old seeds is an exploratory legacy diagnostic, not confirmation. A pass
only authorizes a later protocol whose new DSMC seeds are committed before any
trajectory or outcome exists. A failure stops that compute spend.

The ensemble component alone is not a sufficient novelty claim. The defensible
candidate contribution is the fail-closed workflow: moment-preserving kinetic
reconstruction, development-only convex selection, target-free disagreement
abstention, raw-budget noninferiority gates, immutable prediction hashes, and
fresh-seed confirmation.

## One-line Unity submission

Run this from the Unity project root, normally
`/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc`:

```bash
MV12_TMP="$(mktemp -d)" && git clone --depth 1 --filter=blob:none --sparse --branch agent/mv12-sage-qy https://github.com/Ehsan-Roohi/DSMC_Python.git "${MV12_TMP}/repo" && git -C "${MV12_TMP}/repo" sparse-checkout set vision_guided_dsmc_mv8_kinetic_moments_bundle vision_guided_dsmc_mv9_heat_flux_bundle vision_guided_dsmc_mv10_qy_multiscale_bundle vision_guided_dsmc_mv12_sage_qy_bundle && bash "${MV12_TMP}/repo/vision_guided_dsmc_mv12_sage_qy_bundle/install_and_submit_unity.sh" "$PWD"
```

The installer reads `LAST_MOHAMMADZADEH_MV10_QY_JOB.env`, verifies the
completed MV10/MV9 ancestry, runs the MV12 unit tests, and submits only CPU
prediction/postprocessing jobs. It neither submits nor cancels DSMC jobs.

## Monitor and collect

```bash
source LAST_MOHAMMADZADEH_MV12_SAGE_QY_JOB.env
squeue -j "${MV12_PREDICT_JOB_ID},${MV12_POST_JOB_ID}"
sacct -j "${MV12_PREDICT_JOB_ID},${MV12_POST_JOB_ID}" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS,NodeList
```

The post job writes `MV12_SAGE_QY_ANALYSIS_BUNDLE_*.zip` directly into the
Unity project root and records its path/SHA256 in
`LAST_MOHAMMADZADEH_MV12_SAGE_QY_RESULT.env`.

## Method basis

The design is informed by Noise2Noise restoration from independent noisy
targets, deep-ensemble disagreement under distribution shift, multiple control
variates for kinetic solvers, multi-output approximate control variates, and
selective regression with abstention. The locked protocol records the exact
primary-source URLs and all selection/gating thresholds.
