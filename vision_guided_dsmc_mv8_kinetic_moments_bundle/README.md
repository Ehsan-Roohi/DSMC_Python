# MV8 kinetic-moment feasibility pilot

This bundle creates a new exploratory stage after the locked MV7 temperature
and velocity budget study. It does not modify or rerun MV5, MV6, or MV7.

The completed DSMC checkpoints already contain additive raw moments for every
sampling block. MV8 uses those immutable accumulators to reconstruct, without
an approximation or a new DSMC trajectory:

- shear stress, `Pxy`;
- normal-stress difference, `Pxx-Pyy`;
- horizontal heat flux, `qx`;
- vertical heat flux, `qy`.

Before model submission, an assembly job recursively verifies the source
artifacts, independently rebuilds the stored heat flux, checks the pressure
covariance, and requires the development-only Raw `B=10` estimate to improve
on Raw `B=1`. If that information gate fails, the six neural tasks record a
guarded skip and the postprocessor still returns an audit archive.

If the gate passes, the bundle trains NAFNet-Small and MambaIRv2-Tiny adapted
with three locked initialization seeds at `B=1`, evaluates all four
confirmatory conditions against paired Raw `B=10`, and creates four
publication-quality two-row physical contour figures. Errors are percentages
of fixed `p_ref` or `q_ref`; pointwise division by a locally vanishing stress
or heat flux is explicitly forbidden.

Run from any Unity directory:

```bash
MV8_TMP="$(mktemp -d)" && git clone --depth 1 --filter=blob:none --sparse --branch agent/mv6-reference-stability-repair https://github.com/Ehsan-Roohi/DSMC_Python.git "${MV8_TMP}/repo" && git -C "${MV8_TMP}/repo" sparse-checkout set vision_guided_dsmc_mv8_kinetic_moments_bundle && bash "${MV8_TMP}/repo/vision_guided_dsmc_mv8_kinetic_moments_bundle/install_and_submit_unity.sh"
```

The installer writes `LAST_MOHAMMADZADEH_MV8_KINETIC_JOB.env`. The final job
places an upload-safe ZIP in `$HOME` and prints its path and SHA256.
