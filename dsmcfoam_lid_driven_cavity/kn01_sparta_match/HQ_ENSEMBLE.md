# High-quality Kn=0.1 dsmcFoam ensemble

This production campaign reduces Monte Carlo noise while preserving the
physical model and the 80x80 comparison grid used by job 62947908.

Each of the three members uses:

- Kn = 0.1 and the same argon/wall/lid model as the SPARTA comparison;
- 128 simulator particles per cell (four times the original 32);
- 200,000 time steps at 5e-12 s;
- 40,000 warm-up steps and 160,000 averaging steps;
- approximately 819,200 simulator particles.

The members use 16, 20, and 24 MPI ranks. OpenFOAM-v2406 initializes the DSMC
random generator from the MPI rank, so the different decompositions produce
reproducible decorrelated stochastic trajectories. They should be described as
decomposition-decorrelated members, not as user-specified RNG seeds.

Relative to the original 32-particle, 16,000-sample run, each member contains
approximately 40 times more particle-step sampling. Combining all three gives
approximately 120 times more sampling and an ideal uncorrelated-noise reduction
of sqrt(120), about 11 times. Actual reduction will be smaller because adjacent
DSMC samples are correlated.

Run from a Unity login node:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/agent/dsmcfoam-kn01-production-hq/dsmcfoam_lid_driven_cavity/kn01_sparta_match/hpc/run_unity_hq.sh)
```

The launcher writes `LAST_DSMCFOAM_KN01_HQ_JOB.env` under
`/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK`.

