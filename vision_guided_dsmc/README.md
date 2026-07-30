# Vision-Guided DSMC Pilot

A reproducible research pilot for **vision-guided particle allocation** in a two-dimensional rarefied thermal cavity.

> Stage 4 uses physical SI units, three molecular velocity components, diffuse thermal walls, an Argon VHS model, and SBT/TAS adaptive collision subcells. It remains a research pilot rather than a validated production DSMC solver.

## Install and test

```bash
cd vision_guided_dsmc
python -m pip install -e '.[full]'
pytest -q
```

The Stage-4 module was executed locally with eight focused tests passing. These tests cover VHS scaling, collision conservation, physical cavity execution, conservative reallocation, cell-weight equalization, the diffuse-wall assignment fix, and the high-Knudsen fallback.

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

Reproduce it with:

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

Reproduce the physical benchmark with:

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
- `vgdsmc/training.py`: experimental learned-model path.

## Scientific limitations

- neither solver is yet validated against an independent DSMC implementation;
- variable particle weights are conservatively equalized before SBT but remain an approximate treatment;
- each Stage-4 case currently uses one seed and a four-times-particle reference;
- `Kn < 0.15` is a pilot safeguard, not a general regime boundary;
- wall heat flux, viscosity, collision frequency, and Knudsen-layer profiles still need dedicated validation;
- the reported particle ratios are not wall-clock speedups.

## Next scientific steps

1. validate VHS/SBT relaxation against a homogeneous relaxation and viscosity benchmark;
2. validate the thermal cavity against an independent high-particle DSMC implementation;
3. repeat each `(Kn, temperature ratio)` condition over multiple independent seeds with confidence intervals;
4. add wall heat flux and uncertainty to the vision objective;
5. train continuous score regression and require it to beat the corrected physical baseline.
