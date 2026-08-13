# MV14 KCR-Cavity: kinetic-conservation reconstruction and five-arm ablation

MV14 addresses the persistent wall-normal heat-flux (`q_y`) failure without
using Fourier's law, Newtonian stress, Navier--Stokes closure, or prescribed
wall heat flux. It reuses the immutable MV9 additive DSMC blocks and the
completed MV12 failure record. It does not launch a new DSMC trajectory.

For every B=1 image, MV14 reloads exactly the same indexed raw additive block
and reconstructs `rho`, all three velocity components, the complete symmetric
pressure tensor, `T`, `qx`, and `qy`
directly from molecular velocity moments. The steady monatomic internal-energy
moment of the Boltzmann equation,

```text
div(q) = -(3/2) n k_B u.grad(T) - P:grad(u),
```

is used in weak finite-volume integral form. A generalized least-squares
projection softly balances the direct third-moment observation against these
interval integrals. The first and last cell rows remain observations; no wall
value or additive constant is supplied by a continuum model. Anti-Fourier heat
transfer remains admissible.

The method is not advertised as machine-vision acceleration unless the locked
five-arm ablation shows that the hybrid beats both single-component arms:

1. Raw DSMC B=1;
2. MV9 Mamba vision-only;
3. raw B=1 plus weak kinetic physics only;
4. vision plus weak kinetic physics;
5. paired Raw DSMC B=10.

Development conditions alone select smoothing span and GLS strength. Legacy
test labels are loaded only by a dependency-separated post job after the
prediction archive is SHA256 locked. The old seeds are diagnostic; a successful
outcome only authorizes a separately locked fresh-seed confirmation.

## One-line Unity run

Run from the existing Unity project root:

```bash
MV14_TMP="$(mktemp -d)" && git clone --depth 1 --filter=blob:none --sparse --branch agent/mv14-kinetic-conservation-cavity https://github.com/Ehsan-Roohi/DSMC_Python.git "${MV14_TMP}/repo" && git -C "${MV14_TMP}/repo" sparse-checkout set vision_guided_dsmc_mv8_kinetic_moments_bundle vision_guided_dsmc_mv9_heat_flux_bundle vision_guided_dsmc_mv12_sage_qy_bundle vision_guided_dsmc_mv14_kinetic_conservation_cavity_bundle && bash "${MV14_TMP}/repo/vision_guided_dsmc_mv14_kinetic_conservation_cavity_bundle/install_and_submit_unity.sh" "$PWD"
```

The installer requires completed
`LAST_MOHAMMADZADEH_MV10_QY_JOB.env` and
`LAST_MOHAMMADZADEH_MV12_SAGE_QY_JOB.env`, reuses the MV10/MV9 Torch
environment and raw checkpoints, validates the lock and tests, and submits two
CPU postprocessing jobs.

## Monitor and collect

```bash
source LAST_MOHAMMADZADEH_MV14_KINETIC_CAVITY_JOB.env
squeue -j "${MV14_JOB_IDS}"
sacct -j "${MV14_JOB_IDS}" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS,NodeList
```

The post job writes
`MV14_KINETIC_CONSERVATION_CAVITY_ANALYSIS_BUNDLE_*.zip` and
`LAST_MOHAMMADZADEH_MV14_KINETIC_CAVITY_RESULT.env` to the project root. The
archive includes 600-dpi PNG/PDF qy ablation contours, a full four-field cavity
comparison, condition/seed metrics, weak-balance diagnostics, protocol, and
recursive hashes.
