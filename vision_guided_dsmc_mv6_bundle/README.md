# MV6 locked four-architecture screen

This stage answers one narrow question before any larger sampling-budget
matrix is run: at `budget=1`, is the corrected conditioned U-Net the best
learned reconstruction backbone for the Mohammadzadeh DSMC fields?

It compares four learned residual models:

1. corrected conditioned residual U-Net;
2. NAFNet-Small;
3. MambaIRv2-Tiny adapted;
4. FNO-residual small.

Every model uses the same locked development split, cross-seed targets,
validation set, physical scaling for `log10(Kn)` and `U_lid/100`, bounded
residual head, loss, optimizer, early stopping, batch size, and three training
initialization seeds.  The four architectures contain 62,624--68,418 trainable
parameters (maximum/minimum ratio below 1.10).

The Mamba candidate is explicitly a small dependency-free physical-field
adaptation of the MambaIRv2 ideas, not the official full image-super-resolution
network.  It retains semantic prompts, non-causal four-direction prefix-state
mixing, local depthwise mixing, and gated residual blocks so that it can be
compared fairly at the same parameter scale and run in the existing Unity
PyTorch environment without a custom `mamba_ssm` CUDA build.

## Data and leakage contract

- No new DSMC trajectory is generated.
- The stage waits for and reuses the 16 MV5 confirmatory references.
- Training and validation use only the four MV3 development conditions.
- MV5 confirmatory targets are used only after training, validation selection,
  and baseline selection are complete.
- Raw, validation-selected Gaussian, and validation-selected TSVD/POD are
  reported beside every neural architecture.

## Promotion rule

An architecture is merely marked eligible for a later full matrix when its
mean over the three locked training seeds:

- beats Raw in all four confirmatory conditions; and
- beats the better of Gaussian and TSVD/POD in at least three conditions.

The later `budget=(1,2,5,10)` matrix is **not** submitted automatically.  The
screen ends with a comparison bundle so the next run can be chosen after the
results are inspected.

## Unity execution

Run from the existing `vision_guided_dsmc` project where MV5 was installed:

```bash
bash vision_guided_dsmc_mv6_bundle/install_and_submit_unity.sh "$PWD"
```

The submission creates:

- `moh_mv6_arch`: 12 tasks (`4 architectures x 3 training seeds`), at most four
  concurrent, dependent on the MV5 reference job when it is still running;
- `moh_mv6_post`: aggregate metrics, figures, recursive verification, CSV, and
  the portable return bundle.

Job IDs are written to
`LAST_MOHAMMADZADEH_ARCHITECTURE_SCREEN_JOB.env`.  The default result path is:

```text
results/mohammadzadeh_2012/mv6_architecture_screen
```

The final archive is:

```text
MOHAMMADZADEH_MV6_ARCHITECTURE_SCREEN_BUNDLE.tar.gz
```
