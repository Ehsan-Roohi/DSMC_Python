# Vision-Guided DSMC Pilot

A reproducible research pilot for **vision-guided particle allocation** in a two-dimensional rarefied thermal cavity.

> Stage 4 now uses physical SI units, three translational velocity components, diffuse thermal walls, the Argon VHS model, and SBT/TAS adaptive collision subcells. It is still a research pilot rather than a fully validated production DSMC solver.

## Install and test

```bash
cd vision_guided_dsmc
python -m pip install -e '.[full]'
pytest -q
```

The Stage-4 addition was executed locally with six new physical-kernel tests passing. The full repository test suite should be run through the command above.

## Stage 3: closed-loop physics-vision prototype

The dimensionless pilot established:

- weighted particle states;
- conservative cell-wise splitting and merging;
- exact global particle budgets;
- a reference-free temperature-gradient vision score;
- uniform-versus-adaptive closed-loop continuation;
- an experimental U-Net path whose negative results are documented rather than hidden.

Reproduce the Stage-3 benchmark with:

```bash
vgdsmc-benchmark --output outputs/stage3_benchmark
```

The committed summary is `results/stage3_benchmark_summary.json`.

## Stage 4: physical Argon VHS/SBT cavity

The new physical solver includes:

- SI-unit positions, velocities, time step, cell volume, number density, and mean free path;
- two-dimensional spatial motion with three-dimensional molecular velocities;
- diffuse fully accommodating thermal walls;
- Argon reference parameters `d_ref=4.17e-10 m`, `T_ref=273 K`, and `omega=0.81`;
- VHS total cross section using reduced mass and `Gamma(5/2-omega)`;
- the SBT candidate-pair probability adapted from the repository's `Parallel_TAS.py`;
- adaptive two-dimensional collision subcells;
- conservative equalization of mixed particle weights before SBT collisions;
- confidence-gated vision allocation that falls back to a uniform map when the sampled temperature field is too noisy.

A discrepancy was found in the old standalone kernel: its hard-coded `gamma_val=1.04533` does not equal `Gamma(5/2-0.81)`. Stage 4 uses the evaluated gamma function and the identical-particle reduced mass explicitly.

Reproduce the physical benchmark with:

```bash
vgdsmc-physical-benchmark --output outputs/stage4_physical
```

Verified local Stage-4 result:

- `Kn=0.05`: error reduced by `10.22%` with particle ratio `1.25`;
- `Kn=0.10`: error reduced by `7.20%` with particle ratio `1.25`;
- `Kn=0.20`: the statistical-noise gate disabled adaptation, giving no degradation and particle ratio `1.00`;
- mean error reduction across the three cases: `5.81%`;
- mean particle ratio: `1.167`;
- non-worse cases: `3/3`.

The committed execution record is `results/stage4_physical_summary.json`.

These figures are deterministic pilot results for the specified seeds. They are not yet ensemble confidence intervals or evidence of wall-clock acceleration.

## Code structure

- `vgdsmc/simulator.py`: dimensionless weighted pilot solver;
- `vgdsmc/adaptive.py`: Stage-3 exact-budget reallocation;
- `vgdsmc/vision.py`: Stage-3 reference-free image features;
- `vgdsmc/closed_loop.py`: Stage-3 uniform/adaptive continuation;
- `vgdsmc/vhs_model.py`: physical Argon VHS parameters, cavity configuration, particles, and diffuse walls;
- `vgdsmc/sbt_solver.py`: physical SBT/TAS collision kernel, sampling, and time advancement;
- `vgdsmc/physical_adaptive.py`: confidence gate, physical priority image, and conservative reallocation;
- `vgdsmc/physical_benchmark.py`: reproducible Stage-4 benchmark;
- `vgdsmc/training.py`: experimental U-Net training and inference.

## Scientific limitations

- the physical solver has not yet been validated against an independent DSMC package or an analytical benchmark;
- variable particle weights are locally equalized before SBT, which is conservative but remains an approximate adaptive-weight treatment;
- the current benchmark uses one seed per case and a four-times-particle reference;
- wall heat flux, collision frequency, viscosity, and Knudsen-layer profiles still require dedicated validation;
- the learned vision model has not yet beaten the physics-vision baseline.

## Next scientific steps

1. validate the VHS/SBT relaxation rate against a homogeneous relaxation or viscosity benchmark;
2. validate the thermal cavity against an independent high-particle DSMC implementation;
3. repeat each `(Kn, temperature ratio)` case over multiple independent seeds with confidence intervals;
4. add wall heat-flux and uncertainty-aware objectives to the allocation map;
5. train a continuous score-regression network and require it to beat the confidence-gated physical baseline.
