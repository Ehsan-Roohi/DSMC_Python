# Vision-Guided DSMC Pilot

A reproducible research pilot for **vision-guided particle allocation** in a two-dimensional rarefied thermal cavity.

The physical path uses SI units, two-dimensional particle positions, three molecular velocity components, diffuse fully accommodating walls, an Argon VHS cross-section, and an SBT/TAS collision-selection rule adapted from `Parallel_TAS.py`. It remains a research pilot rather than a validated production DSMC solver.

## Install and test

```bash
cd vision_guided_dsmc
python -m pip install -e '.[full]'
pytest -q
```

GitHub Actions on Python 3.11 verified editable installation, `35 passed in 3.71s`, and a successful end-to-end `vgdsmc-generate` smoke run.

## Critical wall-reflection correction

The early pilot used a NumPy pattern equivalent to:

```python
modify(velocity[particle_ids])
```

Advanced indexing returns a copy, so reflected velocities were not written back to the original particle array. Both the dimensionless and physical solvers now explicitly assign the returned velocities:

```python
velocity[particle_ids] = reflected_velocity
```

Dedicated regression tests verify both wall implementations. The following pre-fix numbers are withdrawn and must not be cited:

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
- cell-wise particle reallocation preserving represented mass, momentum, and energy;
- bounded integer PPC allocation with an exact global particle budget.

## Matched-cost deterministic benchmark

The physical comparison is **vision allocation versus uniform allocation with exactly the same total number of simulation particles**. Both arms undergo the same conservative reallocation.

```bash
vgdsmc-physical-benchmark \
  --output outputs/stage5_physical_equal_budget
```

The original committed three-seed snapshot gave a mean error reduction of `4.72%`, but the multi-seed study below shows that this aggregate was seed-sensitive. It is retained only as a reproducible exploratory snapshot, not as evidence of robust improvement.

Record: `results/stage5_physical_equal_budget_summary.json`.

## Multi-seed matched-cost benchmark

```bash
vgdsmc-physical-multiseed \
  --output outputs/stage6_physical_multiseed \
  --workers 3 \
  --seeds 11 22 33
```

Nine executed runs, with three seeds at each condition:

- equal adaptive/uniform particle budget in every run;
- improved runs: `5/9`;
- overall mean adaptive-to-uniform error ratio: `0.99918`;
- overall mean improvement: `0.08%`;
- normal-approximation 95% interval for the ratio: `[0.9303, 1.0680]`;
- therefore no statistically resolved overall improvement in this small pilot.

Condition-level results:

- `Kn=0.05`: mean ratio `0.99909`; strongly seed-sensitive;
- `Kn=0.10`: mean ratio `1.03042`; strongly seed-sensitive and worse on average;
- `Kn=0.20`: ratios `0.96972`, `0.96080`, `0.97355`; all three seeds improved, with mean reduction `3.20%`.

The honest current conclusion is that the present gradient/noise priority is **not robust across the full tested range**. The `Kn=0.20` condition is a promising subregime that warrants a larger ensemble, not yet a general claim.

Record: `results/stage6_physical_multiseed_summary.json`.

## Independent SBT/VHS collision validation

```bash
vgdsmc-validate-collisions \
  --output outputs/stage5_collision_validation
```

For one fixed Maxwellian velocity sample, the exact pre-clipping expectation is obtained by summing `sigma(g) g` over every unordered pair. Over `5000` independent SBT sweeps:

- exact expected collisions per sweep: `0.50077`;
- measured mean: `0.52700`;
- relative difference: `5.24%`;
- standard error: `0.01016`;
- maximum initial candidate probability: `0.01966`, so clipping is inactive.

A homogeneous anisotropic sample relaxed while conserving total temperature and kinetic energy to numerical precision. Record: `results/stage5_collision_validation_summary.json`.

## Learned-model status

The U-Net path remains experimental and is not yet a successful result:

- single-seed classification collapsed toward the high-refinement class;
- continuous single-seed rank regression had mean Spearman correlation near `0.036`;
- four-member ensemble labels improved it only to about `0.133`.

A learned model must beat the corrected multi-seed matched-cost baseline before it is enabled in the closed loop.

## Main files

- `vgdsmc/simulator.py`: corrected dimensionless weighted pilot solver;
- `vgdsmc/vhs_model.py`: physical VHS parameters, particle state, and corrected diffuse walls;
- `vgdsmc/sbt_solver.py`: physical SBT/TAS collision kernel and advancement;
- `vgdsmc/physical_adaptive.py`: exact-budget physical allocation and conservative reallocation;
- `vgdsmc/physical_benchmark.py`: deterministic matched-cost benchmark;
- `vgdsmc/physical_multiseed.py`: parallel multi-seed benchmark and uncertainty summary;
- `vgdsmc/collision_validation.py`: collision-frequency and homogeneous-relaxation validation;
- `vgdsmc/training.py`: experimental learned-model path.

## Scientific limitations

- no independent DSMC-code validation yet;
- only three seeds per condition;
- the reference uses four times the simulation-particle count rather than a full convergence study;
- variable particle weights are conservatively equalized before SBT collisions but remain approximate;
- equal particle counts do not automatically imply equal wall-clock cost;
- wall heat flux, viscosity, Knudsen-layer profiles, and transport coefficients require dedicated validation.

## Next scientific steps

1. run at least 10-20 independent seeds per condition, especially around `Kn=0.20`;
2. redesign the priority score for `Kn=0.05-0.10`, where the present map is seed-sensitive;
3. validate temperature, density, velocity, and wall heat flux against an independent high-particle DSMC solution;
4. measure particle updates and wall-clock time for matched-error and matched-cost comparisons;
5. train a continuous vision score and require statistically significant improvement over the corrected physical baseline.
