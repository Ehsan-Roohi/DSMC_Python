# MV15A SITR-QY: spectral information audit before new DSMC

MV15A addresses the persistent wall-normal heat-flux (`q_y`) failure as an
estimation problem rather than adding another continuum closure or another
neural architecture. It launches no DSMC trajectory and trains no new network.
It reuses the recursively verified MV9 and MV14 artifacts.

The stage first applies an orthonormal 2-D DCT, appropriate for the
nonperiodic cavity, to matching B1 blocks from distinct development seeds. It
estimates signal power from cross-spectra and sampling-noise power from
half-differences. This produces a mode-wise SNR, a linear Wiener/MMSE bound,
and a block-autocorrelation audit. These diagnostics test whether B1 contains
enough recoverable information to compete with an independent Raw B10.

MV15A then locks a spectral trust-region estimator:

```text
DCT(qy_hat) = w_raw(k) DCT(qy_Raw_B1)
            + (1 - w_raw(k)) DCT(qy_MV9_Mamba).
```

The target-free cross-seed reliability anchors the weights. The strength of
their development-error refinement and radial smoothing are selected by
leaving out one whole development condition at a time. Legacy labels remain
unread until all predictions and controls have been recursively SHA256 locked.

The locked controls are essential: a parametric condition-only field that
never sees B1, and a fusion arm whose Raw-B1 observation is cyclically swapped
with a different condition. MV15A also computes an exact array-level
orthogonal error budget (amplitude, mean offset, and residual), instead of
inferring that budget from separately normalized summary statistics.

Success on old seeds is diagnostic only. Fresh DSMC is authorized only if the
fusion passes every predeclared Raw-B10/seed/condition gate and beats
vision-only, TSVD, condition-only, and the permuted-observation control. A
future claim still requires both fresh seeds and a completely fresh condition.

## One-line Unity run

Run this command from any Unity shell; it explicitly targets the canonical
project and therefore does not depend on a Jupyter/Open OnDemand working
directory:

```bash
MV15A_ROOT=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc; MV15A_TMP="$(mktemp -d)" && git clone --depth 1 --filter=blob:none --sparse --branch agent/mv15a-spectral-information-audit https://github.com/Ehsan-Roohi/DSMC_Python.git "${MV15A_TMP}/repo" && git -C "${MV15A_TMP}/repo" sparse-checkout set vision_guided_dsmc_mv9_heat_flux_bundle vision_guided_dsmc_mv12_sage_qy_bundle vision_guided_dsmc_mv14_kinetic_conservation_cavity_bundle vision_guided_dsmc_mv15a_spectral_information_audit_bundle && bash "${MV15A_TMP}/repo/vision_guided_dsmc_mv15a_spectral_information_audit_bundle/install_and_submit_unity.sh" "${MV15A_ROOT}"
```

The installer requires completed MV10/MV9 and MV14 pointers, reuses the
verified MV10 Torch/SciPy environment, runs eight protocol/numerical tests, and
submits a prediction-lock job followed by a dependency-separated legacy post
job.

## Monitor and collect

```bash
source /project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc/LAST_MOHAMMADZADEH_MV15A_SPECTRAL_AUDIT_JOB.env
squeue -j "$MV15A_JOB_IDS"
sacct -j "$MV15A_JOB_IDS" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS,NodeList
```

The post job writes `MV15A_SPECTRAL_INFORMATION_AUDIT_BUNDLE_*.zip`, its
SHA256, and `LAST_MOHAMMADZADEH_MV15A_SPECTRAL_AUDIT_RESULT.env` to the Unity
project root. The compact archive contains the spectral audit, exact error
decomposition, locked legacy metrics, protocol, recursive manifests, and
publication-quality PNG/PDF figures. Raw datasets and model weights are not
duplicated into the archive.
