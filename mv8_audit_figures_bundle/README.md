# MV8 audit-only physical moment figures

This bundle generates the four requested kinetic-moment figure families from
the already assembled MV8 `dataset.npz`:

- shear stress, `Pxy`;
- normal-stress difference, `Pxx-Pyy`;
- horizontal heat flux, `qx`;
- vertical heat flux, `qy`.

For each of the four confirmatory conditions, it produces one two-row figure
per field in vector PDF and 600-dpi PNG.  The layout and typography follow the
MV7 physical temperature suite.  Columns are Reference, Raw DSMC at `B=1`,
Gaussian at `B=1`, TSVD/POD at `B=1`, and paired Raw DSMC at `B=10`.

The upper row shows the signed physical field.  The lower row shows the signed
difference as a percentage of fixed `p_ref` or `q_ref`; it never divides by a
locally vanishing stress or heat flux.  A separate audit summary visualizes
the B=1/B=10 NRMSE improvement and the two integrity discrepancies.

No neural predictions are included because the locked heat-flux consistency
gate held MV8 before model training.  The heat-flux panels are therefore
clearly marked as audit candidates, not validated model results.

Run from Unity:

```bash
MV8F_TMP="$(mktemp -d)" && git clone --depth 1 --filter=blob:none --sparse --branch agent/mv8-audit-physical-figures https://github.com/Ehsan-Roohi/DSMC_Python.git "${MV8F_TMP}/repo" && git -C "${MV8F_TMP}/repo" sparse-checkout set mv8_audit_figures_bundle && bash "${MV8F_TMP}/repo/mv8_audit_figures_bundle/install_and_submit_unity.sh"
```

The installer writes `LAST_MOHAMMADZADEH_MV8_AUDIT_FIGURES_JOB.env`.

