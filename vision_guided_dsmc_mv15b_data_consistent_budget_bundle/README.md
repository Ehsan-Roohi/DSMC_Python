# MV15B DCIR-QY: data consistency and a disjoint DSMC budget ladder

MV15B is the controlled follow-up to the negative MV15A result.  It launches
no new DSMC trajectory and trains no new neural network.  The ten verified B1
blocks already present for every MV9 seed are grouped without reuse into B1,
B2, B3, and B5 observations; the exact merged Raw B10 field remains the paired
comparator.  B3 uses blocks 0--8 and explicitly leaves block 9 unused.

The method fixes the identifiable failure exposed by MV15A.  It uses a full
two-dimensional DCT reliability map instead of a radial bin, predicts only a
correction to the locked Mamba field, and copies the Raw-Bn DC coefficient
exactly:

```text
DCT(qy_hat) = DCT(qy_Mamba)
             + W_B(k) [DCT(qy_RawBn) - DCT(qy_Mamba)],
W_B(0,0) = 1.
```

The trusted non-DC modes are determined by target-free cross-seed signal and
half-difference noise spectra recomputed independently for every disjoint
budget.  The ideal `N_B1/B` scaling is reported only as a secondary comparison,
so measured block correlation is not silently treated as independence.
Reliability threshold and correction strength are selected only with
development-validation labels.  Legacy targets are not loaded until every
B1/B2/B3/B5 prediction and control is recursively SHA256 locked.

Required controls are Raw Bn, Mamba Bn, DC-only, TSVD Bn, development
condition-only, cross-condition permuted Raw Bn, and exact Raw B10.  A legacy
pass can only authorize a separately locked confirmation campaign; it is never
confirmation.  Fresh seeds at `kn0p1_u400` and a genuinely fresh condition are
still mandatory for a paper claim.

## One-line Unity run

Run from any Unity shell.  Activating `dsmc-gpu` first is recommended; the
installer otherwise reuses the verified MV10 Python environment.

```bash
MV15B_ROOT=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc; MV15B_TMP="$(mktemp -d)" && git clone --depth 1 --filter=blob:none --sparse --branch agent/mv15b-data-consistent-budget-ladder https://github.com/Ehsan-Roohi/DSMC_Python.git "${MV15B_TMP}/repo" && git -C "${MV15B_TMP}/repo" sparse-checkout set vision_guided_dsmc_mv9_heat_flux_bundle vision_guided_dsmc_mv12_sage_qy_bundle vision_guided_dsmc_mv14_kinetic_conservation_cavity_bundle vision_guided_dsmc_mv15a_spectral_information_audit_bundle vision_guided_dsmc_mv15b_data_consistent_budget_bundle && bash "${MV15B_TMP}/repo/vision_guided_dsmc_mv15b_data_consistent_budget_bundle/install_and_submit_unity.sh" "${MV15B_ROOT}"
```

## Monitor and collect

```bash
source /project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc/LAST_MOHAMMADZADEH_MV15B_DATA_CONSISTENT_BUDGET_JOB.env
squeue -j "$MV15B_JOB_IDS"
sacct -j "$MV15B_JOB_IDS" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS,NodeList
```

The post job writes `MV15B_DATA_CONSISTENT_BUDGET_BUNDLE_*.zip`, its SHA256,
and `LAST_MOHAMMADZADEH_MV15B_DATA_CONSISTENT_BUDGET_RESULT.env` to the Unity
project root.  The archive includes budget curves, physical contours, the
mode-wise trust map, locked metrics, protocol, and recursive manifests; model
weights, raw datasets, and prediction arrays are excluded.
