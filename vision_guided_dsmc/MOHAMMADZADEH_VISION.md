# Mohammadzadeh MV1 validated-field vision stage

MV1 trains a residual U-Net to reconstruct the validated bulk observables of
the `Kn=0.05`, `U_lid=100 m/s`, `100x100` Mohammadzadeh cavity from short,
noisy temporal blocks. It reuses the completed M3 calculation and launches no
new DSMC trajectories.

## Scientific contract

- inputs: `T`, `u`, `v`, `rho`, and particle `count`;
- outputs: `T` and `u` only;
- article-facing observables: temperature contours/profiles and macroscopic
  lid slip;
- `qx`, `qy`, and `qz` are excluded from inputs, targets, losses, metrics, and
  the accept/hold decision;
- seeds 91901--91906 train, 91907 validates, and 91908 is touched only once for
  final evaluation;
- each target is the converged M3 mean of the other seven seeds.

The result supports only a single-case denoising claim. It does not demonstrate
generalization to other Knudsen numbers, wall speeds, gases, or geometries.

## Unity

The completed M3 directory must contain `block_fields.npz` and `fields.npz`
for all eight seeds. From `vision_guided_dsmc` run:

```bash
bash scripts/submit_mohammadzadeh_vision_unity.sh
```

Monitor and collect:

```bash
source LAST_MOHAMMADZADEH_VISION_JOB.env
squeue -j "$JOB_ID"
tail -f "logs/moh_mv1_${JOB_ID}.out"
sha256sum -c "$MV1_OUTPUT_ROOT/MOHAMMADZADEH_MV1_RETURN_BUNDLE.tar.gz.sha256"
```

The return bundle contains the checkpoint, held-out numerical arrays, JSON
summary/manifest, and unsmoothed contour/line-profile comparison. No bar chart
is produced.

## MV2: JCP evidence benchmark

MV2 is the preregistered follow-up to the accepted single-case pilot. It still
launches no new DSMC trajectories. Instead, it uses every completed M3 seed as
an independent held-out test once, repeats the comparison at temporal budgets
of 1, 2, 5, and 10 blocks, and compares the residual U-Net with raw averaging,
a validation-selected spatial filter, and a validation-selected truncated-SVD
(POD-type) baseline. The 8 folds by 4 budgets form a 32-task CPU array.

MV2 also closes a subtle weakness in the pilot target construction. Within
each fold, training targets are made only from the six training seeds (with
the current training seed excluded); validation and test targets use the
six-seed training mean. Thus neither held-out converged field can enter model
training or baseline tuning, even indirectly.

The protocol is locked in
`reference_data/mohammadzadeh_2012/mv2_jcp_benchmark_protocol.json`. Heat flux
remains excluded. The aggregate report produces only sampling-budget curves,
contours, and line profiles; it does not produce bar charts.

From `vision_guided_dsmc` on Unity:

```bash
export MV2_M3_ROOT=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc/results/mohammadzadeh_2012/m3_qy_precision
bash scripts/submit_mohammadzadeh_vision_mv2_unity.sh
```

Monitor and collect:

```bash
source LAST_MOHAMMADZADEH_VISION_MV2_JOB.env
squeue -j "$ARRAY_JOB_ID,$POST_JOB_ID"
sha256sum -c "$MV2_OUTPUT_ROOT/MOHAMMADZADEH_MV2_JCP_RETURN_BUNDLE.tar.gz.sha256"
```

Only an MV2 pass authorizes a later cross-condition campaign. Consequently,
additional long DSMC trajectories are not spent before the existing dataset
has established a seed-robust advantage over the locked baselines.
