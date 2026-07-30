# Vision-Guided DSMC Pilot

A small, reproducible research pilot for **vision-guided particle allocation** in a two-dimensional rarefied thermal cavity.

> This remains an educational weighted DSMC-like solver, not a production DSMC implementation. The next major physics step is coupling the workflow to the validated VHS/SBT DSMC kernel.

## Current physics case

- unit-square cavity;
- diffuse thermal walls;
- left wall hotter than the right wall;
- stochastic cell collisions;
- coarse and high-particle reference runs;
- weighted particles after adaptive reallocation;
- exact cell-wise conservation of represented mass, momentum, and kinetic energy during splitting/merging.

## Install and test

```bash
cd vision_guided_dsmc
python -m pip install -e '.[full]'
pytest -q
```

The verified Stage-3 test suite contains eight tests, including:

- simulator and dataset smoke tests;
- arbitrary-size U-Net output tests;
- exact particle-budget tests;
- conservative weighted resampling tests;
- a tiny closed-loop vision-guided run.

## Generate a coarse/reference case

```bash
vgdsmc-generate \
  --output outputs/pilot \
  --nx 24 --ny 24 \
  --ppc 20 --reference-ppc 120
```

## Reproduce the Stage-3 benchmark

```bash
vgdsmc-benchmark --output outputs/stage3_benchmark
```

The benchmark runs three thermal-cavity cases using a reference-free image score based on the robustly normalized temperature-gradient magnitude. Particle allocation uses an exact global budget equal to 1.25 times the uniform coarse-particle count.

Verified local result:

- all three adaptive cases improved over their uniform baselines;
- mean adaptive-to-baseline error ratio: `0.92158`;
- mean error reduction: `7.84%`;
- continuation particle-cost ratio: `1.25`;
- mass and energy conservation errors were near machine precision.

The committed summary is in `results/stage3_benchmark_summary.json`.

## Why the physics-vision baseline comes before learned vision

The first learned experiments were actually executed, but were not successful enough to claim:

- three-class learning collapsed toward the high-refinement class on noisy single-seed labels;
- continuous rank regression on single-seed labels had mean Spearman correlation near `0.036`;
- four-member ensemble labels improved it only to about `0.133`.

The likely reason is that coarse-versus-reference local error is dominated by Monte Carlo label noise. The present temperature-gradient image baseline establishes a reproducible closed-loop target that a learned model must later match or beat.

## Code structure

- `vgdsmc/simulator.py`: weighted particle state, diffuse walls, conservative weighted pair collisions, and sampling;
- `vgdsmc/adaptive.py`: exact-budget allocation and conservative cell-wise particle reallocation;
- `vgdsmc/vision.py`: reference-free image features and physics-vision scores;
- `vgdsmc/closed_loop.py`: uniform versus adaptive continuation and error comparison;
- `vgdsmc/benchmark.py`: reproducible three-case benchmark;
- `vgdsmc/training.py`: experimental U-Net training and inference;
- `results/stage3_benchmark_summary.json`: verified execution summary.

## Next scientific steps

1. replace the pilot collision kernel with the validated VHS/SBT DSMC implementation;
2. define local particle weights consistently with real-particle number and collision probability;
3. generate ensemble-averaged labels over Mach, Knudsen number, wall-temperature ratio, and seed;
4. train a score-regression network and compare it against the temperature-gradient baseline;
5. compare error versus total particle updates, wall heat flux, and uncertainty over multiple independent repetitions.
