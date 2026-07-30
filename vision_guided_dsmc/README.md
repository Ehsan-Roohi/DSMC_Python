# Vision-Guided DSMC Pilot

A reproducible research pilot for **vision-guided computational allocation** in a two-dimensional rarefied thermal cavity.

The physical path uses SI units, two-dimensional particle positions, three molecular velocity components, diffuse fully accommodating walls, an Argon VHS cross-section, and an SBT/TAS collision-selection rule adapted from `Parallel_TAS.py`. It remains a research pilot rather than a validated production DSMC solver.

## Install and test

```bash
cd vision_guided_dsmc
python -m pip install -e '.[full]'
pytest -q
```

The locally assembled source passes `30` tests. GitHub Actions additionally checks editable installation and an end-to-end CLI smoke run on Python 3.11.

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

## Matched-cost results

Vision and uniform controls use exactly the same total number of simulation particles and undergo the same conservative reallocation.

The initial three-case snapshot produced a mean error reduction of `4.72%`, but it was seed-sensitive. The nine-run study is the primary committed result:

```bash
vgdsmc-physical-multiseed \
  --output outputs/stage6_physical_multiseed \
  --workers 3 \
  --seeds 11 22 33
```

- equal adaptive/uniform particle budget in every run;
- improved runs: `5/9`;
- overall mean adaptive-to-uniform error ratio: `0.99918`;
- overall mean improvement: `0.08%`;
- normal-approximation 95% interval: `[0.9303, 1.0680]`;
- no statistically resolved overall improvement.

Further executed diagnostics showed that the apparent condition-level gains were not robust:

- ten additional `Kn=0.20` seeds with the original policy gave mean ratio `1.0190`;
- paired continuation with a two-member reference ensemble gave mean ratio `1.0939`;
- batch-means particle allocation gave mean ratio `1.0172` over nine runs and `1.0417` over ten new `Kn=0.20` seeds;
- vision-guided collision-subcell refinement at equal subcell budget gave mean ratio `1.0582`.

The complete negative-results log is:

```text
results/stage7_to_stage14_diagnostic_log.md
```

**Current conclusion:** DSMC-only gradient, temporal-variance, disagreement, batch-standard-error, and collision-subcell images do not robustly identify where extra computation reduces error. Manual tuning of these noisy features is not scientifically justified.

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
- maximum candidate probability: `0.01966`, so clipping is inactive.

A homogeneous anisotropic sample relaxed while conserving total temperature and kinetic energy to numerical precision. Record: `results/stage5_collision_validation_summary.json`.

## Deterministic-reference adapter

The next path uses a lower-noise DVM/Shakhov or other deterministic kinetic reference instead of noisy coarse-versus-DSMC labels.

Required reference file contract:

```text
reference.npz
  T    # (ny, nx), K
  rho  # (ny, nx), positive density or consistently normalized density
  u    # (ny, nx), m/s
  v    # (ny, nx), m/s
```

Optional arrays such as `qx` and `qy` are preserved. All required fields must be finite, positive where appropriate, and aligned with the DSMC cell-center grid.

Build an ML-ready supervised case:

```bash
vgdsmc-reference-case \
  --coarse-case outputs/coarse/case.npz \
  --reference outputs/dvm/reference.npz \
  --output outputs/supervised/case.npz
```

The adapter:

- validates the DVM/reference field contract and grid alignment;
- computes a dimensionless local error from temperature, velocity magnitude, and density;
- produces a continuous score;
- creates rank-based balanced classes that remain valid even when many cells have tied or zero error;
- stores metadata and class counts beside the generated `NPZ` file.

## Learned-model status

The earlier U-Net path is not claimed as successful:

- single-seed classification collapsed toward the high-refinement class;
- continuous single-seed rank regression had mean Spearman correlation near `0.036`;
- four-member ensemble labels improved it only to about `0.133`.

A learned model must be retrained against deterministic or strongly ensemble-averaged targets and must outperform the equal-budget uniform control over independent seeds.

## Main files

- `vgdsmc/simulator.py`: corrected dimensionless weighted pilot solver;
- `vgdsmc/vhs_model.py`: physical VHS parameters, particle state, and corrected diffuse walls;
- `vgdsmc/sbt_solver.py`: physical SBT/TAS collision kernel and advancement;
- `vgdsmc/physical_adaptive.py`: exact-budget physical allocation and conservative reallocation;
- `vgdsmc/physical_benchmark.py`: deterministic matched-cost benchmark;
- `vgdsmc/physical_multiseed.py`: parallel multi-seed benchmark;
- `vgdsmc/physical_paired_ensemble.py`: paired continuation/reference-ensemble evaluation;
- `vgdsmc/physical_policy_study.py`: policy diagnostics;
- `vgdsmc/collision_validation.py`: collision-frequency and homogeneous-relaxation validation;
- `vgdsmc/reference_adapter.py`: deterministic-reference validation and supervised-label generation;
- `vgdsmc/reference_cli.py`: reference-case command-line interface;
- `vgdsmc/training.py`: experimental learned-model path.

## Scientific limitations

- no independent DVM or external DSMC cavity field has been connected yet;
- the reference used in current comparisons is a higher-particle version of the same solver, not a full convergence study;
- variable particle weights are conservatively equalized before SBT collisions but remain approximate;
- equal particle or subcell budgets do not automatically imply equal wall-clock cost;
- wall heat flux, viscosity, Knudsen-layer profiles, and transport coefficients require independent validation.

## Next scientific steps

1. export one DVM/Shakhov thermal-cavity solution to the documented `NPZ` contract;
2. generate matched coarse DSMC cases on the same grid and operating conditions;
3. verify DVM/DSMC nondimensionalization and field alignment;
4. train continuous score regression against deterministic local error;
5. require statistically significant improvement over the equal-budget uniform control before re-enabling closed-loop vision guidance.
