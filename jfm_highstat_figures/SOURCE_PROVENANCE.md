# Solver provenance

The two production solvers are byte-for-byte copies of the validated kernels
in `jfm_rt05_kn001_heavy_rerun` at Git commit `5595b75b02eff2d23287da6de7b5f1f1a4727694`.
Only command-line arguments select the new particle count, run duration,
temperature ratio, Knudsen number, and seeds.

SHA-256:

- `JFM_hs_dsmc_quarter.py`: `d234a93910403bcf4429ab2b20767a2a7d35349b27dab3016d257abfa5881f6f`
- `JFM_bgk_shakhov_quarter.py`: `c20d1a5fac0ab7e019e695a68e7ddecc64571f63102a8a957f61d766d6b4db55`

No spatial filtering, smoothing, interpolation, or velocity projection is
applied to the quantitative fields.
