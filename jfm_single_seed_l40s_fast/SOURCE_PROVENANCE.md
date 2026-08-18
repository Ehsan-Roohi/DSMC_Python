# Solver provenance

The two production solvers are byte-for-byte copies of the validated JFM
high-statistics bundle from repository commit
`91fa6ad95f146b4d861cc0900bf99e899db146f8`.

- `solver/JFM_hs_dsmc_quarter.py` SHA-256:
  `d234a93910403bcf4429ab2b20767a2a7d35349b27dab3016d257abfa5881f6f`
- `solver/JFM_bgk_shakhov_quarter.py` SHA-256:
  `c20d1a5fac0ab7e019e695a68e7ddecc64571f63102a8a957f61d766d6b4db55`

Only the case table, Slurm routing, number of steps, time-block count, live-output
flags, and single-realization summary logic differ from the earlier bundle.
