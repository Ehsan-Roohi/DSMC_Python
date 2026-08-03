# Validation status

This file separates verified results from production jobs that still need a
CUDA-capable cluster.  A smoke run, a kernel verification, and a published
cavity-profile validation are different levels of evidence.

## Current evidence

| Check | Scope | Status | Evidence |
|---|---|---|---|
| Elastic scattering | Common VHS post-collision kernel | PASS | Pair momentum and translational energy agree before/after scattering to the unit-test tolerance. |
| BT collision rate | SBT, GBT, SSBT, SGBT | PASS | 5,000 Monte Carlo trial lists per model; mean estimated all-pair rate is within 0.1% of the exact rate. |
| Cavity execution | All nine CLI models | PASS (smoke) | CPU runs write finite fields and profiles with zero probability exceedances. |
| Bernoulli time-step guard | SBT/GBT/SSBT/SGBT and TAS variants | PASS | The probability-based bound is part of automatic `dt`; oversized user input or observed probability above one stops the run. |
| Mohammadzadeh profile, coarse CPU | NTC, Kn=0.1, 24x24 | FAIL (diagnostic) | Slip RMSE 0.1213; temperature RMSE 2.536 K. |
| Mohammadzadeh profile, refined CPU | NTC, Kn=0.1, 50x50 | PASS (baseline) | Slip RMSE 0.0507 (gate 0.08); temperature RMSE 1.757 K (gate 2 K); zero probability exceedances. |
| Published 200x200 benchmark | NTC/NTC-PreScan, Kn=0.005/0.05/0.1, repeat seeds | PENDING GPU | Production TOML files and the Unity Slurm launcher are included. |
| Production cavity equivalence | All collision algorithms, repeat seeds | PENDING GPU | Run after the NTC baseline passes, then compare confidence intervals rather than one noisy seed. |
| CUDA execution | Same solver with CuPy backend | PENDING HARDWARE | GPU test is automatic when CUDA/CuPy is available; no CUDA device was present in the development environment. |

The 24x24 to 50x50 refinement reduces the slip error by about 58% and the
temperature error by about 31%.  The 50x50 case passes the repository's
interior-profile gate and is the current CPU baseline.  Its raw temperature
curve is still visibly noisy, and it is not a substitute for the paper's
selected 200x200 grid and repeat-seed study.  The repository therefore does
not label the current result as the final publication reproduction.

## Reproduce the committed kernel evidence

```bash
python -m unittest discover -s tests -v
python scripts/verify_collision_rates.py --samples 5000
```

The numerical rate table is in
`results/collision_rate_verification.csv`; the CPU refinement summary is in
`results/cpu_grid_convergence.csv`.

## Complete the publication-level matrix

First establish the benchmark with the common NTC-PreScan baseline:

```bash
sbatch hpc/unity_gpu_validation.slurm configs/production_mohammadzadeh_kn01.toml
sbatch hpc/unity_gpu_validation.slurm configs/production_mohammadzadeh_kn05.toml
sbatch hpc/unity_gpu_validation.slurm configs/production_mohammadzadeh_kn005.toml
```

Use at least three independent seeds.  Once the reference gate passes, run
`scripts/compare_models.py` at the same physical resolution.  BT-family runs
must keep automatic `dt`; they usually require more time steps to cover the
same physical time.  Report profile differences with repeat-seed confidence
intervals and never infer equivalence from equal random seeds alone.
