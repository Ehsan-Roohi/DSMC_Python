# Bird-QK Gate 3B: live 32-seed cross-language validation

Gate 3B strengthens Gate 3 without changing the verified Fortran or Python
chemistry kernels.

It requires:

- 32 independent Fortran realizations;
- 32 freshly executed Python/Numba realizations;
- no frozen oracle;
- conservation and particle-identity checks in both implementations;
- temperature and species profile equivalence;
- peak-normalized errors for minor species, including OH, H2O, and HO2;
- event-count agreement using relative limits and a three-standard-error
  ensemble criterion;
- rare recombination observed in at least 90% of the runs in each
  implementation, without requiring every low-count realization to be nonzero;
- linearly interpolated OH and H2 induction/consumption locations.

This remains a prescribed post-shock Lagrangian induction test. It is a strong
cross-language verification gate, not a self-consistently resolved reacting
shock or nozzle calculation.

## Unity

Run:

    bash <(curl -fsSL https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/main/qk_gate3b_bundle/submit_qk_gate3b_unity.sh)

The job requests 8 CPU cores, 24 GB of memory, and four hours.
