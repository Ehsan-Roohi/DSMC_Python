# Source provenance

The two GPU solver files are byte-for-byte copies of the production kernels
used for the completed JFM heavy revision runs. They were not rewritten for
this endpoint rerun.

| File | SHA-256 |
| --- | --- |
| `JFM_hs_dsmc_quarter_22m.py` | `d234a93910403bcf4429ab2b20767a2a7d35349b27dab3016d257abfa5881f6f` |
| `JFM_bgk_shakhov_quarter_22m.py` | `c20d1a5fac0ab7e019e695a68e7ddecc64571f63102a8a957f61d766d6b4db55` |

The rerun-specific changes are limited to the nine-row case table, Slurm and
bootstrap scripts, tests, and the ensemble summarizer. The summarizer replaces
the old mean-density cross product with the fully independent estimator in
which every velocity pair is weighted by the density from the unused seed.
