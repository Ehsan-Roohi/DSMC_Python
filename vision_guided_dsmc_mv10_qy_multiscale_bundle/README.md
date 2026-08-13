# MV10 qy multiscale bias-repair pilot

MV10 is the disclosed method-development stage after the completed MV9 pilot.
MV9 passed every provenance/additivity/PSD/software gate, but its locked primary
scientific claim failed because `qy` retained a broad, sign-coherent bias at
`kn0p1_u400`. Mamba repaired `qx` and preserved the stress fields, so MV10 keeps
those three Mamba ensemble channels bitwise unchanged and replaces only `qy`.

The MV10 `qy` model has three explicit paths:

- a dilated local residual trunk for wall and shear-layer structure;
- an eight-cell pooled coarse path for the missing low-frequency amplitude;
- a global-mean path for the coherent sign/magnitude bias.

Its deterministic loss is pixel MSE plus coarse-field, global-mean, and gradient
MSE. It remains a bounded residual model and forbids adversarial, perceptual, and
diffusion losses.

## Scientific status

This stage was designed after the MV9 outcomes were observed. Consequently,
seeds `94301-94304` are explicitly legacy diagnostics and can never be called
confirmatory again. They are not loaded by the model jobs and do not control
training or residual-alpha selection. Postprocessing uses them only to decide
whether a separately locked run with fresh, previously unseen DSMC seeds is
worth the compute budget.

Passing every MV10 diagnostic gate authorizes that fresh-seed stage; it does not
itself establish a JCP result.

## One-line Unity submission

Run from the Unity machine-vision project directory (normally
`/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc`):

```bash
MV10_TMP="$(mktemp -d)" && git clone --depth 1 --filter=blob:none --sparse --branch agent/mv10-qy-multiscale-repair https://github.com/Ehsan-Roohi/DSMC_Python.git "${MV10_TMP}/repo" && git -C "${MV10_TMP}/repo" sparse-checkout set vision_guided_dsmc_mv8_kinetic_moments_bundle vision_guided_dsmc_mv9_heat_flux_bundle vision_guided_dsmc_mv10_qy_multiscale_bundle && bash "${MV10_TMP}/repo/vision_guided_dsmc_mv10_qy_multiscale_bundle/install_and_submit_unity.sh" "$PWD"
```

The installer requires the completed MV9 output referenced by
`LAST_MOHAMMADZADEH_MV9_HEAT_FLUX_JOB.env`. It verifies the MV9 source ancestry,
completed failure decision, dataset manifest, recursive return verification, and
all six model-task manifests before submitting three MV10 model jobs. The
assembly gate scopes the four disclosed legacy seeds to the locked primary
condition and independently verifies that the complete B1/B10 condition and
identity maps agree; seeds belonging to the other three evaluation conditions
are never mistaken for primary-condition identities.

## Monitor and inspect

```bash
source LAST_MOHAMMADZADEH_MV10_QY_JOB.env
squeue -j "${MV10_ASSEMBLE_JOB_ID},${MV10_MODEL_JOB_ID},${MV10_POST_JOB_ID}"
sacct -j "${MV10_ASSEMBLE_JOB_ID},${MV10_MODEL_JOB_ID},${MV10_POST_JOB_ID}" --format=JobID,JobName%24,State,ExitCode,Elapsed,AllocCPUS,MaxRSS,NodeList
python -m json.tool "${MV10_OUTPUT_ROOT}/assembly_summary.json"
python -m json.tool "${MV10_OUTPUT_ROOT}/summary.json"
```

The post job writes a compact timestamped
`MV10_QY_ANALYSIS_BUNDLE_*.zip` directly in the Unity project root and records
its path/SHA256 in `LAST_MOHAMMADZADEH_MV10_QY_RESULT.env`. The ZIP contains the
protocol, summaries, full diagnostic CSV, vector PDF, 600-dpi PNG, Slurm
accounting, and recursive manifest; it excludes the large dataset and weights.

## Publication path

The paper cannot claim the first Noise2Noise method, the first neural Monte Carlo
denoiser, or even the first self-supervised DSMC denoiser. Those directions are
already represented by [Noise2Noise (ICML
2018)](https://proceedings.mlr.press/v80/lehtinen18a.html), [Wei et al.'s
physics-informed dual-sample DSMC denoiser](https://doi.org/10.1016/j.actaastro.2025.12.056),
the [JCP denoising multiscale particle
method](https://doi.org/10.1016/j.jcp.2025.114096), and a [JCP Bayesian
DSMC-CFD surrogate coupling method](https://doi.org/10.1016/j.jcp.2024.113500).

The defensible candidate JCP contribution is instead the full fail-closed
sampling-budget framework: exact additive checkpoint reconstruction of both
second- and third-order kinetic moments, disjoint-seed noisy targets, explicit
provenance/PSD/additivity audits, component-specific low-frequency repair of the
difficult heat-flux direction, raw-budget noninferiority gates, and an abstaining
workflow that requires fresh-seed confirmation before making a claim. The
local-coarse-global CNN block alone is not a sufficient novelty claim.

If MV10 passes, the next protocol must lock new DSMC seeds before their
trajectories or outcomes exist. It should compare the promoted hybrid against
Raw `B=1,2,5,10`, Gaussian, TSVD/POD, and the unmodified MV9 networks, report
condition-clustered uncertainty, and include end-to-end cost rather than only
inference time.
