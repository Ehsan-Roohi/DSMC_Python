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

## MV3: condition-held-out JCP benchmark

MV3 tests the claim that the reconstruction method transfers to a physical
condition that was absent from training.  It reuses the eight completed
`Kn=0.05, U_lid=100 m/s` M3 trajectories and creates only twelve new
references: four seeds each for `(Kn,U_lid)=(0.05,200), (0.05,400),
(0.1,100)`.  The costly `Kn=0.005` and `R200` extensions are deliberately not
authorized at this stage.

Each of the four conditions is held out once, at budgets of 1, 2, 5, and 10
temporal blocks.  The conditioned residual U-Net receives `T,u,v,rho,count`,
`log10(Kn)`, and `U_lid/100`; it reconstructs only `T,u`.  Model training,
validation, baseline selection, and residual-gate selection contain no field
from the held-out condition.  Test targets are leave-one-seed-out means within
the held-out condition.  The locked residual gate may reduce the correction
to zero when validation says a high-budget correction is harmful.

The postprocessor recursively verifies every reference and model artifact,
recomputes all reported metrics from the NPZ arrays, checks the split from the
locked protocol, and only then creates the return bundle.  Figures are line
curves, unsmoothed contours/error maps, and physical profiles with digitized
PRE points where they exist; no bar charts or heat-flux claims are produced.

From `vision_guided_dsmc` on Unity:

```bash
export MV3_M3_ROOT=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc/results/mohammadzadeh_2012/m3_qy_precision
bash scripts/submit_mohammadzadeh_vision_mv3_unity.sh
```

Monitor and collect:

```bash
source LAST_MOHAMMADZADEH_VISION_MV3_JOB.env
squeue -j "$REFERENCE_JOB_ID,$MODEL_JOB_ID,$POST_JOB_ID"
sha256sum -c "$MV3_OUTPUT_ROOT/MOHAMMADZADEH_MV3_JCP_RETURN_BUNDLE.tar.gz.sha256"
```

The twelve DSMC references are a resumable Slurm array (maximum three
concurrent jobs).  They must all pass mechanics and stationarity before the
sixteen condition-held-out model tasks can start.
