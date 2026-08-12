# MV7 locked JCP sampling-budget matrix

This bundle implements the post-MV6 experiment required to turn the
architecture screen into a Journal of Computational Physics sampling-efficiency
study.  It does not add another architecture and it does not regenerate the
sixteen repaired references.

## Locked order of operations

1. Verify the repaired reference tree and the completed twelve-task MV6 screen.
2. Compute Raw, validation-selected Gaussian, and validation-selected TSVD/POD
   for every budget `B = 1, 2, 5, 10`.
3. Only after all four baseline tasks pass, train the four promoted
   architectures for `B = 2, 5, 10` with the three existing initialization
   seeds: exactly 36 new model tasks.
4. Reuse, without retraining, the twelve MV6 `B = 1` tasks.
5. Apply the locked paired non-inferiority and cost analyses and create a small
   return bundle that excludes model weights and prediction arrays.

The full analysis plan is stored in
`mv7_jcp_budget_matrix_analysis_plan.json`.  The primary endpoint is the locked
T-u composite NRMSE.  Each method-budget pair is compared with paired
Raw-at-10 errors on the same sixteen evaluation seeds.  Neural errors are first
averaged over their three training initializations.  The four within-condition
means are then tested using a one-sided 95% Student-t upper bound and a locked
10% relative non-inferiority margin.

## Additions requested before MV7

- classical baselines are present on the complete budget curve;
- the primary endpoint, pairing, margin, confidence rule, and multiplicity
  policy are locked before any new model outcome;
- Raw `B^{-1/2}` scaling, empirical equivalent budget, variance-reduction
  factor, large-budget bias floor, and end-to-end cost equations are reported;
- the generalization claim is explicitly limited because the four evaluation
  conditions informed the earlier MV6 promotion decision;
- a wall-band and spectral error diagnostic examines the FNO weakness at
  `Kn=0.1, U_lid=400 m/s` without claiming that the diagnostic proves a cause;
- Slurm accounting, training/inference timings, parameter counts, checksums,
  seeds, and selected classical hyperparameters are retained.

## Unity one-line submission

Run from any Unity directory:

```bash
TMP="$(mktemp -d)" && git clone --depth 1 --filter=blob:none --sparse --branch agent/mv6-reference-stability-repair https://github.com/Ehsan-Roohi/DSMC_Python.git "$TMP/repo" && git -C "$TMP/repo" sparse-checkout set vision_guided_dsmc_mv7_jcp_bundle && bash "$TMP/repo/vision_guided_dsmc_mv7_jcp_bundle/install_and_submit_unity.sh"
```

The default installed checkout is:

```text
/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc
```

The installer refuses a duplicate submission when
`LAST_MOHAMMADZADEH_MV7_JCP_JOB.env` already exists.  A genuinely new run
requires the explicit environment flag `MV7_ALLOW_NEW_RUN=1`.

The Slurm chain contains:

- `moh_mv7_base`: four cheap baseline tasks;
- `moh_mv7_model`: 36 neural tasks, at most four concurrent;
- `moh_mv7_post`: locked statistics, figures, recursive verification, cost
  accounting, and the lite archive.

The final archive is:

```text
MOHAMMADZADEH_MV7_JCP_BUDGET_MATRIX_LITE.tar.gz
```

Large `model.pt` and `predictions.npz` files remain on Unity and are not copied
into the lite archive, keeping it suitable for the upload-size limit.
