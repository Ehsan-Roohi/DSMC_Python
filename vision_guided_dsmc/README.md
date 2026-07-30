# Vision-Guided DSMC Pilot

A reproducible research pilot for **vision-guided particle allocation** in a two-dimensional rarefied thermal cavity.

> Stage 4 uses physical SI units, three molecular velocity components, diffuse thermal walls, an Argon VHS model, and SBT/TAS adaptive collision subcells. Stage 5 adds direct collision-frequency and homogeneous-relaxation validation. This remains a research pilot rather than a validated production DSMC solver.

## Install and test

```bash
cd vision_guided_dsmc
python -m pip install -e '.[full]'
pytest -q
```

The focused local validation now includes the Stage-4 physical tests and the Stage-5 collision-frequency and relaxation tests. GitHub Actions installs the full optional dependency set so the U-Net shape test is included as well.

## Critical wall-reflection correction

The original pilot used a NumPy pattern equivalent to:

```python
modify(velocity[particle_ids])
```

Advanced/fancy indexing returns a copy, so the reflected velocities were not written back to the original particle array. Both the dimensionless and physical solvers now explicitly assign the returned velocities:

```python
velocity[particle_ids] = reflected_velocity
```

All committed Stage-3 and Stage-4 summaries were regenerated after this correction. Earlier pre-fix benchmark values must not be used.

## Stage 3: corrected dimensionless closed loop

The corrected dimensionless pilot includes weighted particle states, conservative cell-wise splitting/merging, an exact global particle budget, a reference-free temperature-gradient score, and uniform-versus-adaptive continuation.

```bash
vgdsmc-benchmark --output outputs/stage3_benchmark
```

Corrected deterministic rerun using the original three seeds:

- adaptive cases improved: `3/3`;
- case mean-error ratios: `0.89030`, `0.91059`, `0.87616`;
- mean adaptive-to-uniform error ratio: `0.89235`;
- mean error reduction: `10.77%`;
- continuation particle ratio: `1.25`;
- reallocation conservation errors remained near machine precision.

The execution record is `results/stage3_benchmark_summary.json`. This is an educational dimensionless pilot result, not physical DSMC validation.

## Stage 4: physical Argon VHS/SBT cavity

The physical solver includes:

- SI-unit positions, velocities, time step, volume, number density, and mean free path;
- two-dimensional spatial motion with three-dimensional molecular velocities;
- diffuse fully accommodating thermal walls;
- Argon reference parameters `d_ref=4.17e-10 m`, `T_ref=273 K`, and `omega=0.81`;
- VHS total cross section using the identical-particle reduced mass and `Gamma(5/2-omega)`;
- the sequential SBT candidate probability adapted from `Parallel_TAS.py`;
- adaptive two-dimensional collision subcells;
- conservative equalization of mixed particle weights before SBT collisions;
- a temperature/density/noise priority image with an exact global particle budget;
- a conservative fallback to uniform allocation.

A discrepancy was found in the older standalone VHS function: the hard-coded `gamma_val=1.04533` is not `Gamma(5/2-0.81)`. The new physical implementation evaluates the gamma function and uses the reduced mass explicitly. The legacy isolated `vhs_sbt.py` function was corrected as well.

```bash
vgdsmc-physical-benchmark --output outputs/stage4_physical
```

Corrected deterministic Stage-4 result:

- `Kn=0.05`: error reduced by `10.70%`, particle ratio `1.25`;
- `Kn=0.10`: error reduced by `2.40%`, particle ratio `1.25`;
- `Kn=0.20`: the current policy was disabled and remained uniform, error ratio `1.00`;
- improved cases: `2/3`; non-worse cases: `3/3`;
- mean error reduction: `4.37%`;
- mean particle ratio: `1.167`.

The `Kn < 0.15` guard is an empirical no-harm rule derived from this small pilot, not a universal physical threshold. Without that guard, the tested `Kn=0.20` case became worse. The execution record is `results/stage4_physical_summary.json`.

## Stage 5: SBT/VHS kernel validation

The sequential SBT candidate rule is validated independently of the cavity. For a fixed set of Maxwellian velocities, the exact pre-clipping expected collision count is computed by summing `sigma(g) g` over every unordered pair. Thousands of independent SBT sweeps are then compared with that expectation.

```bash
vgdsmc-validate-collisions \
  --output outputs/stage5_collision_validation
```

Verified deterministic result:

- exact expected accepted collisions per sweep: `0.50077`;
- measured mean over `5000` sweeps: `0.52700`;
- relative difference: `5.24%`;
- standard error: `0.01016`;
- maximum initial candidate probability: `0.01966`, so probability clipping is inactive.

A second validation starts from directional temperatures near `(609, 141, 132) K`. After 100 collision sweeps they become approximately `(283, 316, 283) K`:

- anisotropy ratio, final/initial: `0.0707`;
- total-temperature relative change: `5.15e-16`;
- velocity-energy relative change: `0.0` at reported precision;
- maximum mean-velocity change: `1.58e-14 m/s`.

The execution record is `results/stage5_collision_validation_summary.json`. This validates SBT candidate statistics and relaxation/conservation for the selected homogeneous tests; it is not yet a viscosity or transport-coefficient validation.

## Learned-model status

The U-Net path remains experimental and is not yet a successful result:

- single-seed classification collapsed toward the high-refinement class;
- continuous single-seed rank regression had mean Spearman correlation near `0.036`;
- four-member ensemble labels improved it only to about `0.133`.

A learned model must beat the corrected confidence-gated physical baseline before it is enabled in the closed loop.

## Code structure

- `vgdsmc/simulator.py`: corrected dimensionless weighted pilot solver;
- `vgdsmc/adaptive.py`: Stage-3 exact-budget conservative reallocation;
- `vgdsmc/vision.py`: Stage-3 reference-free image features;
- `vgdsmc/closed_loop.py`: Stage-3 uniform/adaptive continuation;
- `vgdsmc/vhs_model.py`: physical VHS parameters, cavity state, and corrected diffuse walls;
- `vgdsmc/sbt_solver.py`: physical SBT/TAS collision kernel and advancement;
- `vgdsmc/physical_adaptive.py`: physical priority image, confidence gate, and reallocation;
- `vgdsmc/physical_benchmark.py`: reproducible Stage-4 benchmark;
- `vgdsmc/collision_validation.py`: Stage-5 pair-expectation and relaxation validation;
- `vgdsmc/training.py`: experimental learned-model path.

## Scientific limitations

- neither cavity solver is yet validated against an independent DSMC implementation;
- variable particle weights are conservatively equalized before SBT but remain an approximate treatment;
- each Stage-4 case currently uses one seed and a four-times-particle reference;
- `Kn < 0.15` is a pilot safeguard, not a general regime boundary;
- wall heat flux, viscosity, and Knudsen-layer profiles still need dedicated validation;
- the reported particle ratios are not wall-clock speedups;
- the Stage-5 frequency test validates candidate statistics for one sampled velocity set, not the full Boltzmann collision integral.

## Next scientific steps

1. estimate viscosity from homogeneous shear/relaxation and compare with the VHS target law;
2. validate the thermal cavity against an independent high-particle DSMC implementation;
3. repeat each `(Kn, temperature ratio)` condition over multiple independent seeds with confidence intervals;
4. add wall heat flux and uncertainty to the vision objective;
5. train continuous score regression and require it to beat the corrected physical baseline.
