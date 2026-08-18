# DSMC component

`scripts/generate_jfm_maxwell_kngu020_case.py` generates either matched
`Kn_Gu = 0.05` or `0.20` input deck despite the historical `020` filename.
The validator locks the 15-field dump schema, VSS `omega = 1`, `alpha = 2.14`,
grid, particle and sampling settings, and reconstructed Gu mean free path.

Example local generation (input only):

```bash
python scripts/generate_jfm_maxwell_kngu020_case.py --seed 104729 --kn-gu 0.20 --output /tmp/k20
python scripts/validate_jfm_maxwell_kngu_case.py /tmp/k20 --kn-gu 0.20
```

For Slurm, submit `hpc/unity_sparta_maxwell_kngu005_020_jfm_single.slurm` as a
two-task array after setting `SPARTA_CASE_ROOT`, `SPARTA_RESULTS_BASE` and
`SPARTA_BIN`. Set `SPARTA_SOURCE_DIR` to the pinned upstream checkout. If the
binary and source tree are separated, export `SPARTA_COMMIT` after independently
verifying it; the driver refuses any value other than the pinned revision.
The run script refuses to overwrite an existing case.

The official SPARTA source is not vendored; see `../THIRD_PARTY_NOTICES.md`.
