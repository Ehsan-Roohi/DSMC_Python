# MV6 publication-figure Unity bundle

This bundle installs and submits only the postprocessing needed to create the
Mohammadzadeh MV6 publication figures. It does not submit DSMC simulations or
model training and does not modify completed task artifacts.

The submitted job creates:

- temperature and signed relative-error maps for all four confirmatory cases;
- horizontal-velocity and lid-speed-normalized error maps for all four cases;
- temperature and velocity profile comparisons;
- composite-NRMSE and validated-profile comparison figures;
- PDF, 400-dpi PNG, CSV, metadata, checksums, and a compact ZIP archive.

Run from any Unity directory:

```bash
TMP="$(mktemp -d)" && git clone --depth 1 --filter=blob:none --sparse --branch agent/mv6-reference-stability-repair https://github.com/Ehsan-Roohi/DSMC_Python.git "$TMP/repo" && git -C "$TMP/repo" sparse-checkout set vision_guided_dsmc_mv6_publication_bundle && bash "$TMP/repo/vision_guided_dsmc_mv6_publication_bundle/install_and_submit_unity.sh"
```

The default target checkout is:

```text
/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc
```

Pass another checkout as the first installer argument if needed.
