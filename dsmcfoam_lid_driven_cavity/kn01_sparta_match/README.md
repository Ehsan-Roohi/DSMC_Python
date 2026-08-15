# dsmcFoam cavity at Kn=0.1 for comparison with SPARTA

This case is the `Kn=0.1` companion to the completed SPARTA production case.
It uses OpenFOAM `dsmcFoam`; it does not run SPARTA.

## Matched physical model

- square cavity: `L = 1e-6 m`
- monatomic argon: `mass = 6.63e-26 kg`
- VHS: `diameter = 4.17e-10 m`, `omega = 0.81`, `Tref = 273 K`
- all walls fully diffuse at `300 K`
- top wall velocity: `(100 0 0) m/s`
- `Kn = 0.1`, with `lambda = 1/(sqrt(2)*pi*d^2*n)`
- number density: `1.294383653016e25 1/m3`

## dsmcFoam numerical setup

- `80 x 80 x 1` collision cells; periodic depth `1.25e-8 m`
- `dx/lambda = 0.125`
- target population: approximately 32 simulator particles per cell
- `nEquivalentParticles = 0.7900290850925`
- `deltaT = 5e-12 s`
- 20,000 steps; averaging begins after 4,000 steps
- 16 MPI ranks on one Unity AVX-512 CPU node

The numerical grid and time window are intentionally inherited from the already
completed dsmcFoam `Kn=0.05` run. Solver-to-solver comparisons should interpolate
both solutions onto a common grid and retain raw data. Publication-level claims
still require grid, timestep, particle-count, averaging-window, and seed studies.

## One-line Unity submission

Run on the Unity login node; do not load OpenFOAM there:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/agent/dsmcfoam-kn01-sparta-match/dsmcfoam_lid_driven_cavity/kn01_sparta_match/hpc/run_unity.sh)
```

The launcher writes the submitted job information to:

```text
/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK/LAST_DSMCFOAM_KN01_JOB.env
```

