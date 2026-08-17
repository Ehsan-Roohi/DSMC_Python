# SPARTA JFM rerun at the common Gu Knudsen number

This workflow runs the transition case at the same convention used by the
R13/R26 calculations:

- `Kn_Gu = lambda_Gu/L = 0.20`
- `Kn_VHS(collision diameter) = 0.15435398118188065`
- `n0 = 8.634602561095979e24 m^-3`
- `fnum = 1.317535791182858e6` for `N=160`, `PPC=256`

The submission uses two seeds, 40,000 warm-up steps, 200,000 sampling steps,
a sampling stride of 10, and 20,000 accumulated samples per cell. The build
job first checks the pinned SPARTA source and runs a corrected-convention smoke
case. Every array member independently verifies the Knudsen convention,
density, and `fnum` before launching SPARTA.

On Unity, submit the complete build-array-collector chain with:

```bash
ROOT=/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK/DSMC_Python_sparta_kngu020_jfm; if [ -d "$ROOT/.git" ]; then git -C "$ROOT" fetch origin agent/sparta-kn020-jfm && git -C "$ROOT" switch agent/sparta-kn020-jfm && git -C "$ROOT" pull --ff-only origin agent/sparta-kn020-jfm; else git clone --depth 1 --branch agent/sparta-kn020-jfm --single-branch https://github.com/Ehsan-Roohi/DSMC_Python.git "$ROOT"; fi && bash "$ROOT/sparta_cavity_mohammadzadeh/hpc/bootstrap_unity_sparta_kngu020_jfm.sh"
```

This form uses Git smart HTTP rather than `raw.githubusercontent.com`, avoiding
the shared-IP Raw endpoint rate limit that can return HTTP 429 on compute login
nodes.

The workflow uses a separate checkout
`DSMC_Python_sparta_kngu020_jfm`, stores raw results under
`results/run8_dsmc_kngu020_sparta`, and writes the final return bundle as
`SPARTA_KNGU020_JFM_<array-job-id>.tar.gz`.
