# Vision-Guided DSMC Pilot

A reproducible research pilot for **vision-guided particle allocation** in a two-dimensional rarefied thermal cavity.

The physical path uses SI units, two-dimensional particle positions, three molecular velocity components, diffuse fully accommodating walls, an Argon VHS cross-section, and an SBT/TAS collision-selection rule adapted from `Parallel_TAS.py`. It remains a research pilot rather than a validated production DSMC solver.

## Install and test

```bash
cd vision_guided_dsmc
python -m pip install -e '.[full]'
pytest -q
```

Verified by GitHub Actions on Python 3.11:

```text
35 passed in 3.64s
vgdsmc-generate smoke run: passed
```

## Critical wall-reflection correction

The early pilot used a NumPy pattern equivalent to:

```python
modify(velocity[particle_ids])
```

Advanced indexing returns a copy, so reflected velocities were not written back to the original particle array. Both the dimensionless and physical solvers now explicitly assign the returned velocities:

```python
velocity[particle_ids] = reflected_velocity
```

Dedicated regression tests verify that both wall implementations modify the original particle state. The following pre-fix numbers are withdrawn and must not be cited:

- the earlier `7.84%` dimensionless reduction;
- the earlier `5.81%` physical reduction.

## Corrected dimensionless closed loop

The original three dimensionless cases were rerun after correcting the wall reflection:

- improved cases: `3/3`;
- case mean-error ratios: `0.89030`, `0.91059`, `0.87616`;
- mean adaptive-to-uniform error ratio: `0.89235`;
- mean error reduction: `10.77%`;
- continuation particle ratio: `1.25`;
- conservative-reallocation errors remained near machine precision.

Run:

```bash
vgdsmc-benchmark --output outputs/stage3_benchmark
```

The execution record is `results/stage3_benchmark_summary.json`. This remains an educational dimensionless result, not a physical DSMC validation.

## Physical Argon VHS/SBT solver

The physical solver includes:

- SI-unit positions, velocities, time step, volume, number density, and mean free path;
- two-dimensional spatial motion with three-dimensional molecular velocities;
- diffuse fully accommodating thermal walls;
- Argon reference parameters `d_ref=4.17e-10 m`, `T_ref=273 K`, and `omega=0.81`;
- VHS total cross-section using the identical-particle reduced mass and `Gamma(5/2-omega)`;
- the sequential SBT candidate probability adapted from `Parallel_TAS.py`;
- adaptive two-dimensional collision subcells;
- conservative equalization of mixed particle weights before SBT collisions;
- cell-wise particle reallocation preserving represented mass, momentum, and energy;
- bounded integer PPC allocation with a mathematically exact global particle budget.

The older isolated VHS helper was corrected as well: it previously used molecular mass where the identical-particle reduced mass is required, while an older standalone script hard-coded a gamma value inconsistent with `Gamma(5/2-0.81)`.

## Matched-cost vision benchmark

The primary physical comparison is **vision allocation versus uniform allocation with exactly the same total number of simulation particles**. Both arms undergo the same conservative reallocation, so the comparison isolates where particles are placed rather than merely adding particles.

Run:

```bash
vgdsmc-physical-benchmark \
  --output outputs/stage5_physical_equal_budget
```

Deterministic pilot result using the committed three cases:

- `Kn=0.05`: adaptive is `1.11%` worse than equal-budget uniform;
- `Kn=0.10`: adaptive reduces error by `12.63%`;
- `Kn=0.20`: adaptive reduces error by `2.64%`;
- improved cases: `2/3`;
- all three cases are within `2%` of or better than equal-budget uniform;
- mean adaptive-to-uniform error ratio: `0.95278`;
- mean error reduction: `4.72%`;
- adaptive-to-uniform particle ratio: exactly `1.0`;
- both methods use `800` particles during continuation in each case.

The execution record is `results/stage5_physical_equal_budget_summary.json`. This is a promising deterministic pilot, not yet a publication-grade statistical claim.

## Independent SBT/VHS collision validation

Run:

```bash
vgdsmc-validate-collisions \
  --output outputs/stage5_collision_validation
```

For one fixed Maxwellian velocity sample, the exact pre-clipping expectation is obtained by summing `sigma(g) g` over every unordered pair. Over `5000` independent SBT sweeps:

- exact expected collisions per sweep: `0.50077`;
- measured mean: `0.52700`;
- relative difference: `5.24%`;
- standard error: `0.01016`;
- maximum initial candidate probability: `0.01966`, so probability clipping is inactive.

A homogeneous anisotropic sample relaxed from directional temperatures near `(609, 141, 132) K` to `(283, 316, 283) K` after 100 sweeps:

- final/initial anisotropy ratio: `0.0707`;
- total-temperature relative change: `5.15e-16`;
- velocity-energy relative change: `0.0` at reported precision;
- maximum mean-velocity change: `1.58e-14 m/s`.

The execution record is `results/stage5_collision_validation_summary.json`. These tests validate the selected candidate statistics and conservative homogeneous relaxation, not viscosity or the full Boltzmann collision integral.

## Vision policy

The reference-free physical priority image combines:

- smoothed temperature-gradient magnitude;
- smoothed density-gradient magnitude;
- temporal temperature variation.

The continuous image is converted into a bounded integer PPC map with an exact global budget. The earlier empirical `Kn < 0.15` gate was removed because the fair matched-cost comparison showed improvement in the tested `Kn=0.20` case.

## Learned-model status

The U-Net path remains experimental and is not yet a successful result:

- single-seed classification collapsed toward the high-refinement class;
- continuous single-seed rank regression had mean Spearman correlation near `0.036`;
- four-member ensemble labels improved it only to about `0.133`.

A learned model must beat the corrected matched-cost physical baseline before it is enabled in the closed loop.

## Main files

- `vgdsmc/simulator.py`: corrected dimensionless weighted pilot solver;
- `vgdsmc/adaptive.py`, `vision.py`, `closed_loop.py`: dimensionless allocation loop;
- `vgdsmc/vhs_model.py`: physical VHS parameters, particle state, and corrected diffuse walls;
- `vgdsmc/sbt_solver.py`: physical SBT/TAS collision kernel and advancement;
- `vgdsmc/physical_adaptive.py`: exact-budget physical allocation and conservative reallocation;
- `vgdsmc/physical_benchmark.py`: matched-cost physical benchmark;
- `vgdsmc/collision_validation.py`: collision-frequency and homogeneous-relaxation validation;
- `vgdsmc/training.py`: experimental learned-model path.

## Scientific limitations

- the cavity has not yet been validated against an independent DSMC implementation;
- variable particle weights are conservatively equalized before SBT collisions but remain an approximate treatment;
- the matched-cost benchmark currently uses one seed per condition and a four-times-particle reference;
- one matched-cost case is `1.11%` worse;
- equal particle counts do not automatically imply equal wall-clock cost;
- wall heat flux, viscosity, Knudsen-layer profiles, and transport coefficients require dedicated validation;
- the collision-frequency test covers one sampled velocity set, not the full Boltzmann collision integral.

## Next scientific steps

1. repeat every `(Kn, temperature ratio)` condition over multiple independent seeds and report confidence intervals;
2. validate temperature, density, velocity, and wall heat flux against an independent high-particle DSMC solution;
3. estimate viscosity or homogeneous relaxation rates and compare with the VHS target law;
4. measure particle updates and wall-clock time for matched-error and matched-cost comparisons;
5. train a continuous vision score and require statistically significant improvement over the corrected physics baseline.
