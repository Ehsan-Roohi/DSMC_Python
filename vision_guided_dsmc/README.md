# Vision-Guided DSMC Pilot

A reproducible research pilot for **vision-guided particle allocation** in a two-dimensional rarefied thermal cavity.

The current physical path uses SI units, two-dimensional particle positions, three molecular velocity components, diffuse fully accommodating walls, an Argon VHS cross-section, and an SBT/TAS collision-selection rule adapted from `Parallel_TAS.py`. It remains a research pilot rather than a validated production DSMC solver.

## Install and test

```bash
cd vision_guided_dsmc
python -m pip install -e '.[full]'
pytest -q
```

Verified local execution of the current branch:

```text
24 passed
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

Dedicated regression tests now verify that both wall implementations modify the original particle state.

The following pre-fix numbers are withdrawn and must not be cited:

- the earlier `7.84%` dimensionless reduction;
- the earlier `5.81%` physical reduction.

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
- cell-wise particle reallocation preserving represented mass, momentum, and energy.

## Matched-cost vision benchmark

The scientifically relevant comparison is **vision allocation versus uniform allocation with exactly the same total number of simulation particles**. Both arms undergo the same conservative reallocation operation, so the comparison does not confuse vision allocation with the effect of merely adding particles.

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

The execution record is:

```text
results/stage5_physical_equal_budget_summary.json
```

This is a promising pilot result, not yet a publication-grade statistical claim. Each condition currently uses one seed and a four-times-particle reference.

## Vision policy

The reference-free physical priority image combines:

- smoothed temperature-gradient magnitude;
- smoothed density-gradient magnitude;
- temporal temperature variation.

The resulting continuous image is converted into an exact global particles-per-cell budget. The current policy is deterministic and physics-based. The earlier empirical `Kn < 0.15` gate has been removed because the fair matched-cost comparison showed improvement in the tested `Kn=0.20` case.

## Learned-model status

The U-Net path remains experimental and is not yet a successful result:

- single-seed classification collapsed toward the high-refinement class;
- continuous single-seed rank regression had mean Spearman correlation near `0.036`;
- four-member ensemble labels improved it only to about `0.133`.

A learned model must beat the corrected matched-cost physical baseline before it is enabled in the closed loop.

## Code structure

- `vgdsmc/simulator.py`: corrected dimensionless weighted pilot solver;
- `vgdsmc/adaptive.py`: dimensionless exact-budget conservative reallocation;
- `vgdsmc/vision.py`: dimensionless reference-free image features;
- `vgdsmc/closed_loop.py`: dimensionless continuation workflow;
- `vgdsmc/vhs_model.py`: physical VHS parameters, particle state, and corrected diffuse walls;
- `vgdsmc/sbt_solver.py`: physical SBT/TAS collision kernel and advancement;
- `vgdsmc/physical_adaptive.py`: physical priority image and conservative reallocation;
- `vgdsmc/physical_benchmark.py`: matched-cost physical benchmark;
- `vgdsmc/training.py`: experimental learned-model path.

## Scientific limitations

- the cavity has not yet been validated against an independent DSMC implementation;
- variable particle weights are conservatively equalized before SBT collisions but remain an approximate treatment;
- the matched-cost benchmark currently uses only one seed per condition;
- wall heat flux, viscosity, Knudsen-layer profiles, and transport coefficients require dedicated validation;
- equal particle counts do not automatically imply equal wall-clock cost;
- the current reference uses four times as many simulation particles, not a formal grid/time/particle convergence study.

## Next scientific steps

1. repeat every `(Kn, temperature ratio)` condition over multiple independent seeds and report confidence intervals;
2. validate temperature, density, velocity, and wall heat flux against an independent high-particle DSMC solution;
3. estimate viscosity or homogeneous relaxation rates and compare with the VHS target law;
4. measure particle updates and wall-clock time for matched-error and matched-cost comparisons;
5. train a continuous vision score and require statistically significant improvement over the corrected physics baseline.
