# Audited Vision-Guided DSMC / DVM Pilot

A reproducible rarefied-gas research pipeline for testing whether spatial information from a coarse DSMC solution can guide computational allocation in a two-dimensional thermal cavity.

The repository now contains:

- a physical Argon VHS/SBT DSMC pilot with 2-D positions and 3-D molecular velocities;
- corrected diffuse fully accommodating walls;
- deterministic BGK-DVM and corrected three-velocity Shakhov-DVM references;
- raw DVM/Tecplot import and aligned `NPZ` reference contracts;
- matched-budget particle, collision-subcell, and temporal-sampling experiments;
- multi-seed learned error-map models;
- explicit negative-result records;
- spatial- and velocity-grid convergence studies for the deterministic Shakhov solver.

This is an audited research pilot, not a validated production DSMC package.

## Install and test

```bash
cd vision_guided_dsmc
python -m pip install -e '.[full]'
pytest -q
```

## Critical corrections

### Wall-reflection writeback

The early particle solver passed an advanced-indexing expression such as `velocity[ids]` into a mutating helper. NumPy returned a copy, so reflected velocities were not written back. Both particle solvers now explicitly assign returned velocities:

```python
velocity[ids] = reflected_velocity
```

Dedicated regression tests protect both implementations. Results produced before this correction, including the early `7.84%` and `5.81%` reductions, are withdrawn.

### Exact bounded budgets

Particle and sampling allocators now guarantee the exact requested global integer budget while respecting cell-wise lower and upper limits.

### Shakhov velocity-quadrature energy

A low-order Cartesian velocity grid did not reproduce the continuous Maxwellian second moment exactly and injected energy during relaxation. The Maxwellian parameter is now reconstructed on the active quadrature so an isothermal cavity remains near its prescribed temperature. The corrected solver reports both physical temperature and the raw quadrature diagnostic.

## Physical Argon VHS/SBT path

The physical DSMC pilot includes:

- SI-unit position, velocity, time step, number density, and cell volume;
- Argon `d_ref=4.17e-10 m`, `T_ref=273 K`, `omega=0.81`;
- VHS cross-section using identical-particle reduced mass and `Gamma(5/2-omega)`;
- sequential SBT/TAS collision selection adapted from `Parallel_TAS.py`;
- adaptive two-dimensional collision subcells;
- conservative mass/momentum/energy particle resampling;
- collision-frequency and homogeneous-relaxation checks.

The collision-statistics check obtained an exact expected count of `0.50077` collisions per sweep and a measured mean of `0.52700` over 5000 sweeps. Homogeneous anisotropic relaxation conserved total temperature, momentum, and kinetic energy to numerical precision.

## Deterministic reference path

Generate a corrected Shakhov reference:

```bash
vgdsmc-shakhov-reference \
  --output outputs/dvm/shakhov_reference.npz \
  --nx 12 --ny 12 --nv 10 \
  --kn 0.10 --max-steps 1800
```

Import an existing raw moment file:

```bash
vgdsmc-import-dvm \
  --input data/cavity_dvm/cavity_dvm_moments.dat \
  --output outputs/dvm/imported_reference.npz
```

The aligned reference contract requires:

```text
T, rho, u, v
```

and preserves optional fields such as `qx` and `qy`.

Build an ML-ready DSMC/DVM case:

```bash
vgdsmc-reference-case \
  --coarse-case outputs/coarse/case.npz \
  --reference outputs/dvm/shakhov_reference.npz \
  --output outputs/supervised/case.npz
```

## Audited adaptation findings

All principal comparisons used paired controls and exact equal computational budgets for the quantity under study.

### Particle and collision allocation

The early apparent gains did not survive independent seeds. The confirmatory Stage 19 experiment fixed the selected policy before execution:

- `Kn=0.10`;
- temperature differences `20, 40, 60 K`;
- 5% local particle perturbation;
- 10 entirely new seeds;
- 30 paired equal-particle runs.

Result:

```text
mean adaptive/uniform error ratio = 1.01614
95% interval = [0.97607, 1.05621]
improved runs = 11/30
improving temperature conditions = 0/3
```

The confirmatory hypothesis failed. Learned particle reallocation is disabled.

### Fixed-trajectory sampling allocation

Stages 20 and 21 kept particle trajectories and collisions identical and changed only the number of temporal observations used in each cell, with exactly 720 observations in every estimator.

The best Stage 21 policy used field-error weights and lag-one autocorrelation correction:

```text
mean adaptive/uniform sampling-error ratio = 0.99575
95% interval = [0.97890, 1.01259]
improved runs = 14/30
```

The effect was small and statistically unresolved; `Kn=0.10` worsened. Sampling allocation is also disabled.

### Current scientific conclusion

Gradient maps, temporal variance, two-run disagreement, batch standard error, learned low-frequency scores, particle reallocation, collision-subcell allocation, and fixed-trajectory sampling allocation did not demonstrate robust matched-budget benefit in the tested cavity pilot. These negative results are retained rather than tuned away.

Detailed records are stored under `results/stage7_to_stage14_diagnostic_log.md` and `results/stage15_...` through `results/stage21_...`.

## Stage 22: corrected Shakhov-DVM convergence

Run:

```bash
vgdsmc-dvm-convergence \
  --output-dir outputs/stage22_dvm_convergence \
  --kn 0.10 \
  --spatial-levels 6 8 10 --spatial-reference 12 --spatial-nv 8 \
  --velocity-levels 6 8 10 --velocity-reference 12 --velocity-grid 8
```

The test used `T_left=340 K`, `T_right=260 K`, and `T_top=T_bottom=300 K`.

### Spatial grid at fixed `Nv=8`

Errors relative to `12x12, Nv=8`:

| Grid | T RMS | rho RMS | velocity RMS | heat-flux RMS | composite |
|---|---:|---:|---:|---:|---:|
| 6x6 | 0.6267% | 0.5611% | 0.2671% | 12.8211% | 3.5902% |
| 8x8 | 0.3421% | 0.3000% | 0.1398% | 7.1745% | 2.0013% |
| 10x10 | 0.1510% | 0.1300% | 0.0578% | 3.2333% | 0.8987% |

Every metric decreased monotonically.

### Velocity grid at fixed `8x8`

Errors relative to `8x8, Nv=12`:

| Nv | T RMS | rho RMS | velocity RMS | heat-flux RMS | composite |
|---|---:|---:|---:|---:|---:|
| 6 | 0.1613% | 0.5313% | 0.1049% | 34.9348% | 8.9174% |
| 8 | 0.0822% | 0.1021% | 0.0711% | 2.2977% | 0.6378% |
| 10 | 0.0276% | 0.0356% | 0.0236% | 0.3459% | 0.1080% |

Every metric again decreased monotonically. Heat flux is much more sensitive than bulk fields.

### Practical reference choice

For subsequent physical comparisons:

- use at least `Nv=10`;
- use at least a `12x12` spatial grid when wall or volume heat flux matters;
- run a combined `12x12, Nv=10/12` check before declaring the final reference;
- still compare against an independent solver or published benchmark because Stage 22 is internal convergence, not external validation.

Record: `results/stage22_dvm_convergence_summary.json`.

## Main modules

- `vgdsmc/vhs_model.py`: physical state, VHS parameters, corrected walls;
- `vgdsmc/sbt_solver.py`: VHS/SBT collisions and physical advancement;
- `vgdsmc/collision_validation.py`: collision statistics and relaxation;
- `vgdsmc/dvm_bgk.py`: deterministic BGK-DVM baseline;
- `vgdsmc/dvm_shakhov.py`: corrected Shakhov transport/collision kernel;
- `vgdsmc/dvm_shakhov_corrected.py`: saved reference interface and diagnostics;
- `vgdsmc/dvm_import.py`: raw DVM/Tecplot moment importer;
- `vgdsmc/reference_adapter.py`: aligned deterministic supervision;
- `vgdsmc/dvm_convergence.py`: Stage 22 convergence study;
- `vgdsmc/lowfreq_closed_loop.py`: paired learned particle-allocation audit;
- `vgdsmc/sampling_allocation.py`: fixed-trajectory sampling audit;
- `vgdsmc/effective_sampling_allocation.py`: weighted/autocorrelated sampling audit.

## Remaining limitations and next step

- Shakhov-DVM has internal convergence evidence but no independent external validation;
- heat flux remains the most demanding quantity;
- variable-weight SBT treatment is approximate;
- no wall-clock speedup has been demonstrated;
- adaptive policies remain disabled.

The next scientific step is a combined high-resolution Shakhov reference followed by centerline and wall-heat-flux comparison against an independent DSMC/DVM implementation or a published benchmark. Only after that validation should any new learned allocation strategy be reconsidered.
