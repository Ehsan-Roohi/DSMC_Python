# MV7 JCP publication-figure and cost-closure Unity bundle

This postprocessing-only bundle turns the recursively verified MV7 budget
matrix into the final publication figure suite. It does not submit DSMC
references, retrain a model, alter `summary.json`, or replace any locked MV7
artifact.

The submitted job creates:

- four two-row physical-temperature contour figures, one for every locked
  condition. The upper row shows the actual temperature field for Reference,
  budget-one Raw/Gaussian/TSVD/four neural models, and Raw@B=10; the lower row
  shows signed relative error in percent;
- a two-panel sampling-accuracy and Raw-equivalent-efficiency figure;
- a four-panel non-inferiority forest figure with the locked one-sided 95%
  upper bounds and 10% margin;
- a dedicated large-budget bias-floor figure;
- an FNO diagnostic based on absolute boundary/interior MSE and radial error
  spectra, avoiding a misleading boundary/interior-ratio-only presentation;
- a CPU benchmark that closes the missing reused-MV6 `B=1` inference timing;
  a completed timing closure from the same immutable MV7 root is reused by
  default so a figure-only rerun does not repeat the long benchmark;
- vector PDF, 600-dpi PNG, CSV/JSON provenance, SHA256 checksums, and a compact
  ZIP archive.

Run from any Unity directory:

```bash
TMP="$(mktemp -d)" && git clone --depth 1 --filter=blob:none --sparse --branch agent/mv6-reference-stability-repair https://github.com/Ehsan-Roohi/DSMC_Python.git "$TMP/repo" && git -C "$TMP/repo" sparse-checkout set vision_guided_dsmc_mv7_publication_bundle && bash "$TMP/repo/vision_guided_dsmc_mv7_publication_bundle/install_and_submit_unity.sh"
```

The default target checkout is:

```text
/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc
```

The installer requires `LAST_MOHAMMADZADEH_MV7_JCP_JOB.env` from the completed
MV7 chain. It refuses a duplicate publication submission unless
`MV7_PUBLICATION_ALLOW_NEW_RUN=1` is intentionally supplied.

The return archive is written to `$HOME` as:

```text
MOHAMMADZADEH_MV7_JCP_PUBLICATION_FIGURES_<UTC>.zip
```
