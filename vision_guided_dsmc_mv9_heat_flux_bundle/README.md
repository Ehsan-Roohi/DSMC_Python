# MV9 heat-flux Noise2Noise pilot

MV9 is a prospectively locked recovery stage after the immutable MV8 audit.
It does not rerun DSMC and does not modify the MV5-MV8 protocols or results.

The MV8 model gate failed because its audit converted normalized heat flux to
`float32` before redimensionalizing it for comparison with the stored physical
field. The observed `5.78e-8` discrepancy is the expected size of that
single-precision round trip. MV9 reconstructs and compares stored `qx/qy`
entirely in `float64`; only post-gate neural tensors are cast to `float32`.
The locked `1e-10` provenance tolerance is unchanged.

If provenance passes, MV9 trains NAFNet-Small and MambaIRv2-Tiny with three
initializations. A one-block input from seed `s` is paired only with the
full-window mean from other same-condition seeds. Thus input and target
particle histories are disjoint. The linear Noise2Noise objective prioritizes
`qx`, `qy`, and the wall layer without adversarial, perceptual, or diffusion
losses that could hallucinate a plausible-looking kinetic field.

The stage remains fail-closed:

- source hashes and additive block/full identities are recursively checked;
- pressure covariance must remain positive semidefinite;
- stored/reconstructed heat flux must pass the unchanged `1e-10` gate;
- failed assembly produces an audit-only archive and guarded model skips;
- confirmatory seeds never control preprocessing, training, or selection.

The primary success rule is heat-flux specific: at least one architecture must
match or beat Raw `B=10` in mean `qx/qy` NRMSE at the locked primary condition,
neither component may exceed `1.10 x Raw@B=10`, and the all-moment composite
ratio must not exceed `1.10`.

Run from any Unity directory:

```bash
MV9_TMP="$(mktemp -d)" && git clone --depth 1 --filter=blob:none --sparse --branch agent/mv9-heat-flux-noise2noise https://github.com/Ehsan-Roohi/DSMC_Python.git "${MV9_TMP}/repo" && git -C "${MV9_TMP}/repo" sparse-checkout set vision_guided_dsmc_mv8_kinetic_moments_bundle vision_guided_dsmc_mv9_heat_flux_bundle && bash "${MV9_TMP}/repo/vision_guided_dsmc_mv9_heat_flux_bundle/install_and_submit_unity.sh" "$PWD"
```

The installer writes `LAST_MOHAMMADZADEH_MV9_HEAT_FLUX_JOB.env`. The post job
creates a recursively verified upload-safe ZIP in `$HOME` and prints its path
and SHA256.
