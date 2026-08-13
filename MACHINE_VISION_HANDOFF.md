# DSMC Machine-Vision Project Handoff

Last updated: 2026-08-13 UTC

This document is the durable handoff for the prior ChatGPT conversation titled
`بهبود نتایج ماشین ویژن`. It records the current scientific state, immutable
decisions, repository ancestry, Unity paths, execution command, and the next
required checks. When this document conflicts with a locked protocol, verified
result artifact, or committed source file, the committed protocol/artifact/source
is authoritative.

## 1. Current state at a glance

- Repository: `Ehsan-Roohi/DSMC_Python`
- Stable MV5-MV8 ancestry branch: `agent/mv6-reference-stability-repair`
- Open draft ancestry PR: `#22`, `Add locked MV5 repairs and MV7 JCP budget matrix`
- MV8 commits:
  - `8e4beb382ad8d5a40923853202b2cb2748012b95` — add MV8 pilot
  - `7d03fa92563633cd555e87c14e2cde458ea27210` — fix MV8 additive-moment audit
- Current MV9 branch: `agent/mv9-heat-flux-noise2noise`
- Open draft MV9 PR: `#24`, `Add MV9 heat-flux Noise2Noise pilot`
- MV9 code commit: `65b0c6fc03ed6a769fc293c8c71bdd22c15c457c`
- MV9 protocol status: `locked_before_any_MV9_heat_flux_model_outcome`
- Immediate action: run the locked MV9 Unity command, then inspect the assembly
  gate before interpreting or reporting any model result.

No MV9 model outcome is recorded in the repository at this handoff point. Do not
describe MV9 as successful or failed until the Unity return archive and its
recursive verification are available.

## 2. Scientific objective and scope

The primary project is machine-vision restoration/denoising of DSMC flow fields.
The high-fidelity kinetic/DVM or long-window DSMC solution is a teacher/reference,
not the central research claim by itself.

The present question is whether a one-block DSMC estimate can be restored to the
quality of a ten-block raw estimate while preserving physically meaningful
kinetic moments. The final claim must be supported by locked quantitative gates,
not by visually smoother contours alone.

Current restored outputs are:

- shear stress, `Pxy`;
- normal-stress difference, `Pxx-Pyy`;
- horizontal heat flux, `qx`;
- vertical heat flux, `qy`.

Auxiliary model inputs are normalized density, velocity components, temperature,
`log10(Kn)`, and normalized lid speed.

## 3. Durable locations

### Unity production/reference code

Original reference/teacher project:

```text
/project/pi_roohie_umass_edu/CavityColdToHotIdentify
```

Machine-vision working tree used by MV7-MV9:

```text
/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc
```

Verified MV7 publication/budget source root:

```text
/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc/results/mohammadzadeh_2012/mv7_jcp_budget_matrix/run_20260811T222214Z
```

MV9 defaults to the machine-vision working tree above. Runtime source roots are
read from `LAST_MOHAMMADZADEH_MV7_JCP_JOB.env`; do not guess or silently replace
them.

### Repository bundles

```text
vision_guided_dsmc_mv8_kinetic_moments_bundle/
vision_guided_dsmc_mv9_heat_flux_bundle/
```

The locked MV9 protocol is:

```text
vision_guided_dsmc_mv9_heat_flux_bundle/payload/reference_data/mohammadzadeh_2012/mv9_heat_flux_noise2noise_protocol.json
```

## 4. Stage history and decisions

### Early prototype

The first repository prototype implemented a DSMC-like thermal-cavity pilot,
four-channel `T,u,v,sigma_T` input, allocation labels, and a small U-Net. It was
explicitly educational rather than research-grade. Its role was to establish the
pipeline, not to support the final paper claim.

### MV3: conditioned-vision stress test

MV3 was verified and reproducible, but it failed its preregistered success gates.
The conditioned vision model had an unbounded response on the held-out
`Kn=0.10, U=100 m/s` stress test while Gaussian-like and TSVD/POD baselines
remained stable. MV3 was therefore demoted to an ablation; it is not positive
evidence for a vision-model claim.

### MV5: reference stability repair

Three reference seeds with failed, predeclared in-scope temperature/velocity
stationarity checks were rerun: `94003`, `94201`, and `94301`. The physical
configuration, grid, particles, 3,000 samples, ten blocks, 93,750-step sampling
horizon, and `|z| <= 2` gate remained unchanged; only burn-in moved later.

Original reference directories remained immutable. A separately audited
13-original + 3-repair reference tree was assembled and recursively verified.
Heat-flux `qy` diagnostics were outside the MV5/MV6 `T/u` scope and could not veto
those references.

### MV6: four-model budget-one screen

All three reference repairs, reference assembly, and twelve model tasks completed.
The first aggregate postprocessor failed only because it compared nested derived
floating-point metric dictionaries with exact equality across heterogeneous CPU
nodes. Six shared arrays were exactly identical across all twelve tasks. The
portable recovery retained exact equality for arrays and used
`rtol=1e-6, atol=1e-12` only for derived metric trees.

### MV7: locked JCP sampling-budget matrix

MV7 evaluated Raw DSMC, validation-selected Gaussian, TSVD/POD, and promoted
neural architectures across `B=1,2,5,10`. It reused the completed MV6 `B=1`
models and added 36 tasks for budgets 2, 5, and 10.

Locked primary comparison:

- endpoint: paired `T-u` composite NRMSE;
- comparator: `Raw@B=10`;
- uncertainty: one-sided 95% condition-clustered upper confidence bound;
- non-inferiority margin: `1.10`.

MV7 status is `complete_MV7_reused_budget_one_inference_timing_closure`.
The training-only break-even is a lower bound because shared training-data
generation cost is excluded. FNO spatial/radial diagnostics are descriptive and
do not prove spectral bias or a periodic-boundary cause.

### MV8: immutable kinetic-moment feasibility audit

MV8 reconstructed `Pxy`, `Pxx-Pyy`, `qx`, and `qy` from immutable additive
checkpoint accumulators without a new DSMC trajectory.

Audit outcome:

- all reconstructed fields were finite;
- pressure covariance/PSD check passed;
- block/full additivity passed with maximum reported discrepancy
  `5.34e-14`;
- stored/reconstructed heat-flux provenance failed with discrepancy
  `5.78e-8` against the locked `1e-10` tolerance;
- all six planned neural tasks were guarded skips;
- MV8 remained audit-only and produced no neural kinetic-moment outcome.

The failure was traced to an implementation precision boundary: normalized heat
flux was reconstructed in `float64`, cast to `float32`, and only then
redimensionalized for comparison with stored physical heat flux. The discrepancy
is consistent with that single-precision round trip. MV8 results and protocols
remain immutable.

### MV9: prospectively locked heat-flux recovery pilot

MV9 repairs only the provenance comparison:

- normalization, redimensionalization, and comparison stay in `float64`;
- the tolerance remains exactly `1e-10`;
- conversion to `float32` occurs only after every pre-model gate passes;
- MV5-MV8 protocols, source artifacts, and results are not changed.

MV9 uses cross-seed Noise2Noise with disjoint particle histories. A one-block
input from seed `s` is paired with the full-window mean from other same-condition
training seeds. Confirmatory seeds do not control preprocessing, selection, or
training.

Locked model matrix:

- architectures: `nafnet_small`, `mambairv2_tiny_adapted`;
- initialization seeds: `2608091`, `2608092`, `2608093`;
- total tasks: 6;
- input budget: `B=1`;
- raw comparator: `B=10`;
- primary condition: `kn0p1_u400`;
- representative contour: evaluation seed `94302`, block `0`.

The loss is deterministic and linear: weighted MSE plus gradient MSE. Channel
weights for `Pxy`, `Pxx-Pyy`, `qx`, and `qy` are `[0.75, 0.75, 1.5, 2.0]`;
gradient weight is `0.1`. Adversarial, perceptual, and diffusion losses are
forbidden because plausible-looking hallucinated kinetic fields are unacceptable.

## 5. MV9 fail-closed gates

Model training is authorized only if all applicable assembly gates pass:

1. source checkpoint hashes match the locked ancestry;
2. block/full additive moments satisfy fixed-scale relative
   `L-infinity <= 1e-9`;
3. every reconstructed field is finite;
4. pressure covariance satisfies the PSD tolerance `1e-10`;
5. stored/reconstructed physical heat flux satisfies relative tolerance `1e-10`;
6. on development/validation data, Raw `B=10` improves the composite over
   Raw `B=1` and improves at least three individual fields;
7. confirmatory outcomes do not control the gate.

If assembly fails, the correct result is an audit-only archive plus six guarded
task skips. Do not bypass the gate, relax a tolerance, or submit model tasks
manually.

## 6. Locked MV9 success rule

At least one architecture must satisfy all three conditions at the primary
condition:

1. mean `qx/qy` composite NRMSE ratio to paired `Raw@B=10` is `<= 1.00`;
2. neither `qx` nor `qy` NRMSE ratio to paired `Raw@B=10` exceeds `1.10`;
3. all-four-moment composite NRMSE ratio to paired `Raw@B=10` is `<= 1.10`.

Passing this pilot does not establish a final confirmatory claim. It authorizes a
separately locked full confirmatory follow-up. Failure means that one-block
signal-to-noise and/or model bias does not support a kinetic-moment reduction
claim under this design.

## 7. Exact Unity submission command

Run from the intended Unity working directory:

```bash
MV9_TMP="$(mktemp -d)" && git clone --depth 1 --filter=blob:none --sparse --branch agent/mv9-heat-flux-noise2noise https://github.com/Ehsan-Roohi/DSMC_Python.git "${MV9_TMP}/repo" && git -C "${MV9_TMP}/repo" sparse-checkout set vision_guided_dsmc_mv8_kinetic_moments_bundle vision_guided_dsmc_mv9_heat_flux_bundle && bash "${MV9_TMP}/repo/vision_guided_dsmc_mv9_heat_flux_bundle/install_and_submit_unity.sh" "$PWD"
```

The installer:

- installs the exact locked MV8 source ancestry needed by MV9;
- compiles and verifies the lock;
- submits assembly, a six-task model array, and postprocessing with dependencies;
- writes `LAST_MOHAMMADZADEH_MV9_HEAT_FLUX_JOB.env`;
- refuses a duplicate run unless `MV9_ALLOW_NEW_RUN=1` is deliberately set.

Do not set `MV9_ALLOW_NEW_RUN=1` merely because a job is pending or because its
status has not yet been checked.

## 8. Monitoring and result retrieval

After submission:

```bash
source LAST_MOHAMMADZADEH_MV9_HEAT_FLUX_JOB.env
squeue -j "${MV9_ASSEMBLE_JOB_ID},${MV9_MODEL_JOB_ID},${MV9_POST_JOB_ID}"
sacct -j "${MV9_ASSEMBLE_JOB_ID},${MV9_MODEL_JOB_ID},${MV9_POST_JOB_ID}" \
  --format=JobID,JobName%24,State,ExitCode,Elapsed,AllocCPUS,MaxRSS,NodeList
```

Inspect the assembly decision before the neural tasks or figures:

```bash
python -m json.tool "${MV9_OUTPUT_ROOT}/assembly_summary.json"
```

The post job must create:

- `summary.json`;
- `verification.json`;
- `artifact_manifest.json`;
- `source_moment_audit.csv`;
- `mv9_primary_condition_metrics.csv` when training is authorized;
- vector PDFs and 600-dpi PNGs for all four moments when training completes;
- a recursively verified upload-safe ZIP in `$HOME`, with printed SHA256.

## 9. Interpretation and publication guardrails

- Never equate visual smoothness with accuracy.
- Never use local percentage error divided by a near-zero stress or heat flux.
  Figure errors use a fixed physical reference scale.
- Never modify MV8 to make its failed gate pass. MV9 is the disclosed recovery.
- Never claim that FNO diagnostics prove a specific causal failure mechanism.
- Never report a training-only break-even as total end-to-end cost.
- Never use confirmatory seeds for preprocessing, model selection, or gate tuning.
- Preserve failed, skipped, and audit-only outcomes; they are part of the scientific
  record.

## 10. Required handoff update after MV9 finishes

Append the following without rewriting the locked history:

- assembly, model-array, and postprocessing job IDs;
- exact `MV9_OUTPUT_ROOT`;
- Slurm states and exit codes;
- assembly decision and every failed/passed gate;
- per-architecture locked ratios for heat flux, individual `qx/qy`, and all moments;
- final pilot decision from `summary.json`;
- return ZIP path, size, SHA256, and verified-file count;
- whether a separately locked confirmatory stage is authorized.

## 11. Evidence hierarchy

Use this order when reconstructing project state in a future chat:

1. locked JSON protocols and source hashes;
2. recursively verified result artifacts and manifests;
3. committed source and bundle README files;
4. Slurm accounting and immutable logs;
5. this handoff document;
6. conversational recollection.

This ordering prevents chat-length limits or partial memory retrieval from
changing the scientific record.

## 12. MV9 completed outcome and MV10 handoff (appended 2026-08-13 UTC)

MV9 finished successfully as software and physics-audit execution, but failed
its locked scientific feasibility rule. Preserve both facts.

- Jobs: assembly `62848201`, model array `62848202_[0-5]`, post `62848203`.
- Every job completed with exit code `0:0`.
- Output root:
  `/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc/results/mohammadzadeh_2012/mv9_heat_flux_noise2noise/run_20260813T022352Z`.
- Block/full additivity: `5.3420642483461805e-14` versus locked `1e-9`.
- Heat-flux provenance: `1.8512365465180513e-16` versus locked `1e-10`.
- Minimum PSD eigenvalue ratio: `0.4490867716985249`.
- Recursive return verification: `14` files, decision `verified`.
- Return archive: `MOHAMMADZADEH_MV9_HEAT_FLUX_NOISE2NOISE_20260813T050529Z.zip`,
  `39,136,606` bytes, SHA256
  `e7e213cf9668469cb016340dbc5ad946f2622838a942b6198c44c2c3817301c6`.

At primary condition `kn0p1_u400`, Raw `B=10` qy NRMSE was
`0.0818394756`. NAFNet qy was `0.1345279922` (`1.6438 x Raw B10`) and
Mamba qy was `0.1488360010` (`1.8186 x Raw B10`). Mamba did beat Raw `B=10`
for qx (`0.9195 x`) and both stress components, but its all-moment composite
ratio was `1.2640` and heat-flux composite ratio was `1.4641`. NAFNet's
corresponding ratios were `1.2712` and `1.4510`. Both locked pilot decisions
were false. Official decision:
`MV9_feasibility_does_not_support_one_block_kinetic_moment_claim`.

The qy failure was systematic across all four old evaluation seeds and appeared
as a broad coherent amplitude/sign bias, not merely high-frequency pixel noise.
Those old seeds (`94301-94304`) have now been observed and can never be used as
confirmatory evidence again.

MV10 is therefore an explicitly post-MV9 exploratory repair. Its bundle is
`vision_guided_dsmc_mv10_qy_multiscale_bundle/`. It preserves the MV9 Mamba
stress/qx channels and replaces only qy with a bounded local-coarse-global
residual ensemble trained and selected exclusively on development data. The old
MV9 evaluation set is read only by postprocessing as a legacy diagnostic. Even
if every MV10 diagnostic gate passes, a separate protocol with fresh unobserved
DSMC seeds must be locked and run before any JCP confirmatory claim.

Exact MV10 Unity command:

```bash
MV10_TMP="$(mktemp -d)" && git clone --depth 1 --filter=blob:none --sparse --branch agent/mv10-qy-multiscale-repair https://github.com/Ehsan-Roohi/DSMC_Python.git "${MV10_TMP}/repo" && git -C "${MV10_TMP}/repo" sparse-checkout set vision_guided_dsmc_mv8_kinetic_moments_bundle vision_guided_dsmc_mv9_heat_flux_bundle vision_guided_dsmc_mv10_qy_multiscale_bundle && bash "${MV10_TMP}/repo/vision_guided_dsmc_mv10_qy_multiscale_bundle/install_and_submit_unity.sh" "$PWD"
```

## 13. MV11 independent cylinder geometry (appended 2026-08-13 UTC)

MV11 introduces an external hypersonic geometry rather than another cavity
condition. The locked Bird/VHS argon cylinder state is Mach 10 with
`D=0.3048 m`, `T_inf=200 K`, and `T_wall=500 K`. The default development gate
uses a 194x100 grid, 1.5 million simulator particles, and four prospectively
locked fresh seeds. Legacy seeds from the Shojaa/JFM and MV9 campaigns are
explicitly forbidden.

The bundle `vision_guided_dsmc_mv11_ds2v_cylinder_bundle/` patches the corrected
DS2V source already stored on Unity. It adds restart-safe RNG control when
restart calls exist, guards exact-zero RNG logarithms, and writes thirteen
additive raw moments needed for `Pxy`, `Pxx-Pyy`, `qx`, and `qy`. The complete
Bird source is not duplicated in GitHub; source and executable hashes are
recorded with every campaign.

The post job places a compact `MV11_DS2V_CYLINDER_ANALYSIS_BUNDLE_*.zip`
directly in the Unity project root. MV11 is a prospective second-geometry
data-acquisition gate, not by itself a confirmatory JCP result.
