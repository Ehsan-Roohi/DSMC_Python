# MV11: independent DS2V cylinder geometry

MV11 adds the second geometry to the vision-guided DSMC program: a Bird/VHS
Mach-10 argon flow over a two-dimensional circular cylinder. It is deliberately
different from the closed thermal cavity used in MV8--MV10: the cylinder adds a
bow shock, stagnation region, curved diffuse wall and wake.

## Locked campaign

- corrected Mach-10 state: `n_inf=4.247e20 m^-3`, `T_inf=200 K`,
  `U_inf=2634.1 m/s`, `D=0.3048 m`, `T_wall=500 K`;
- Bird baseline: nearest-neighbour pairing, VHS rate and elastic scattering;
- default development grid `194 x 100`, with `1,500,000` initial simulator
  particles;
- four prospectively locked, previously unused fresh seeds;
- no restart state is shared between seeds;
- thirteen additive raw moments per cell and sampling block;
- reconstruction of `Pxy`, `Pxx-Pyy`, `qx` and `qy` from the raw moments.

The full Bird source is not copied into this public repository. The fail-closed
patcher operates on the corrected source already present in the Unity
`ABINITIO_SHOCK_TESTS_v2` project, records its SHA256, and refuses an
unrecognized source layout. Patcher version 2 also preserves Bird's
`REAL(KIND=8)` interface for simulation time when writing MV11 moment blocks.

## One-line Unity submission

This command can be run from any Unity directory:

```bash
MV11_TMP="$(mktemp -d)" && git clone --depth 1 --filter=blob:none --sparse --branch agent/mv11-ds2v-cylinder https://github.com/Ehsan-Roohi/DSMC_Python.git "${MV11_TMP}/repo" && git -C "${MV11_TMP}/repo" sparse-checkout set vision_guided_dsmc_mv11_ds2v_cylinder_bundle && bash "${MV11_TMP}/repo/vision_guided_dsmc_mv11_ds2v_cylinder_bundle/install_and_submit_unity.sh"
```

The launcher submits one preparation/build job, a four-task seed array (two
simultaneous cases by default), and one dependent analysis/package job.

## Monitor

```bash
source /project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc/LAST_MV11_DS2V_CYLINDER_JOB.env && squeue -j "$MV11_JOB_IDS"
```

## Returned artifact

The post job puts `MV11_DS2V_CYLINDER_ANALYSIS_BUNDLE_*.zip` directly in the
root of the Unity machine-vision project and writes its path and SHA256 to
`LAST_MV11_DS2V_CYLINDER_RESULT.env`. Large raw trajectories remain under the
timestamped campaign directory and are excluded from the compact return ZIP.

## Optional overrides

Set variables before the one-line command when a larger production run is
wanted, for example:

```bash
export MV11_NX=800 MV11_NY=300 MV11_PARTICLES=12000000 MV11_ARRAY_CONCURRENCY=1
```

The default 194x100/1.5M campaign is the fast second-geometry development gate.
The 800x300/12M setting should be reserved for the promoted high-fidelity
reference after this gate is inspected.

## Scientific guardrail

The seeds are locked in `mv11_cylinder_protocol.json` before any MV11 outcome
exists. Once results are inspected they become observed data; method changes
after inspection require a new development/confirmation split. Completion of
this data-acquisition campaign alone is not a JCP confirmation.
