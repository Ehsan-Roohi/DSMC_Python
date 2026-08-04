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
