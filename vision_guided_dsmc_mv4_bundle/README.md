# Mohammadzadeh MV4 stability-repair bundle

This bundle installs an isolated MV4 stage into an existing
`vision_guided_dsmc` Unity checkout. It does not modify MV3 results.

MV4 keeps the locked MV3 data split and T/u targets and adds:

- a residual head bounded by `4*tanh(latent/4)` in training-residual units;
- validation-only residual-amplitude and classical-fallback selection;
- a target-free coordinate-support gate using `log10(Kn)` and `U_lid/100`;
- exact Raw-identity fallback for extrapolation, with Gaussian-like and
  TSVD/POD retained as reported baselines;
- fixed physical sanity bounds and explicit clipping diagnostics;
- a pre-outcome protocol locked to the verified MV3 summary SHA-256;
- recursive metric/artifact verification and physical field figures.

## One-line Unity install and submit

Run this from the existing Unity project root:

```bash
MV4_TMP="$(mktemp -d)" && git clone --depth 1 --filter=blob:none --sparse --branch agent/mv4-stability-repair https://github.com/Ehsan-Roohi/DSMC_Python.git "${MV4_TMP}/repo" && git -C "${MV4_TMP}/repo" sparse-checkout set vision_guided_dsmc_mv4_bundle && bash "${MV4_TMP}/repo/vision_guided_dsmc_mv4_bundle/install_and_submit_unity.sh" "$PWD"
```

The submitter reuses the verified MV3 reference tree and the existing PyTorch
environment. It creates:

```text
LAST_MOHAMMADZADEH_VISION_MV4_JOB.env
results/mohammadzadeh_2012/mv4_safe_reconstruction/
```

No new DSMC reference trajectories are submitted. The model stage contains 16
CPU tasks (`4 folds x 4 budgets`) with at most four concurrent tasks, followed
by one dependent aggregate/verifier/package job.

## Monitor

```bash
source LAST_MOHAMMADZADEH_VISION_MV4_JOB.env
squeue -j "$MV4_MODEL_JOB_ID,$MV4_POST_JOB_ID"
```

## Acceptance contract

The aggregate decision passes only if all 16 task safety checks pass, all
coordinate extrapolations use the Raw-identity fallback, the safe method
never exceeds `1.05 x Raw`, and the trusted one-block cases retain genuine
improvement with at least one win over both selected classical baselines.
