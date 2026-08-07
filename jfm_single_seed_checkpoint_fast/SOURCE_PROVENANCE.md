# Solver provenance

The transport, wall, collision, random-number, sampling-kernel, precision, and
reconstruction code comes from the validated JFM production solvers in commit
`0904a6dc2f7b6aabfe1bcd6aefe3d19640fe4265`:

- original HS SHA-256:
  `d234a93910403bcf4429ab2b20767a2a7d35349b27dab3016d257abfa5881f6f`
- original BGK/Shakhov SHA-256:
  `c20d1a5fac0ab7e019e695a68e7ddecc64571f63102a8a957f61d766d6b4db55`

The only solver changes add the `--checkpoint-steps` interface and a read-only
snapshot writer. The snapshot synchronizes the GPU, copies the existing
float64 moment accumulators, reconstructs raw fields, and writes them without
changing particle state, RNG state, collision state, accumulators, or sampling
counts. The run then continues normally to its final target.

Checkpoint-fast solver SHA-256 values:

- `solver/JFM_hs_dsmc_quarter.py`:
  `3d9f9ff5162d9d7ac18078c99c623f8af67434692a88da76ffdcfcaa83386c31`
- `solver/JFM_bgk_shakhov_quarter.py`:
  `1abbd9d67e7333171146030ccc17963371fb33dd1882b29acc25400004b81ca3`
