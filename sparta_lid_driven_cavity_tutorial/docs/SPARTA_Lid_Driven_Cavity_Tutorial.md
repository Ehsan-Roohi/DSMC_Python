# Running a Rarefied Lid-Driven Cavity with SPARTA

## Purpose

This tutorial demonstrates a complete SPARTA workflow: defining the molecular
model, generating a dimensional input deck, building SPARTA, running serial and
MPI jobs, extracting macroscopic fields, combining independent random seeds,
and reporting the result. The example is self-contained and does not require an
external reference dataset.

Code and reproducibility files:
<https://github.com/Ehsan-Roohi/DSMC_Python/tree/agent/sparta-lid-driven-cavity-tutorial/sparta_lid_driven_cavity_tutorial>

## Physical and numerical setup

The domain is a square cavity with side length `L = 1.0e-6 m`. The stationary
walls and translating lid are fully diffuse at 300 K. The lid moves in the
positive x direction at 100 m/s. Monatomic argon is represented with VHS
parameters `d_ref = 4.17e-10 m`, `omega = 0.81`, `T_ref = 273 K`, and
`alpha = 1.0`. In SPARTA, the collision command uses the `vss` style; setting
`alpha = 1.0` selects the VHS limit.

For `Kn = 0.1`, the mean free path is `lambda = Kn L = 1.0e-7 m`. The number
density is calculated as

```text
n = 1 / (sqrt(2) pi d_ref^2 lambda)
  = 1.294383653e25 m^-3.
```

The production grid is `200 x 200`, so `Delta x/lambda = 0.05`. With 32
simulator particles per cell, the initial population is 1,280,000. The time
step is `8.256788187e-13 s`, or approximately `0.00292` mean collision times.

## The SPARTA input deck

The domain and Cartesian grid are defined by:

```text
dimension            2
boundary             s s p
create_box           0.0 1.0e-6 0.0 1.0e-6 -0.5 0.5
create_grid          200 200 1
```

The gas, stationary walls, and moving diffuse lid are specified by:

```text
species              argon.species Ar
mixture              gas Ar nrho 1.294383653e25 temp 300
surf_collide         fixed diffuse 300 1.0
surf_collide         lid diffuse 300 1.0 translate 100 0.0 0.0
bound_modify         xlo xhi ylo collide fixed
bound_modify         yhi collide lid
collide              vss gas argon.vss
```

The production run uses 14,000 warm-up steps followed by 26,000 sampling
steps. Number density, velocity, and temperature are accumulated every ten
steps:

```text
run                  14000
reset_timestep       0
compute              flow grid all gas nrho u v w temp
fix                  flowavg ave/grid all 10 1 10 c_flow[*] ave running
dump                 fields grid all 26000 grid.final.* id xc yc f_flowavg[*]
run                  26000
```

## Building and testing SPARTA

Clone the tutorial branch and enter the case directory:

```bash
git clone --branch agent/sparta-lid-driven-cavity-tutorial --single-branch \
  https://github.com/Ehsan-Roohi/DSMC_Python.git
cd DSMC_Python/sparta_lid_driven_cavity_tutorial
```

Build the serial or MPI executable and run the fast tests:

```bash
bash scripts/install_sparta_linux.sh serial
python3 -m unittest discover -s tests -v
bash scripts/run_case.sh smoke serial
```

The classroom-sized case is run with:

```bash
bash scripts/run_case.sh tutorial serial
```

## Production execution

A production seed can be run locally with MPI:

```bash
SEED=20260803 MPI_RANKS=16 bash scripts/run_case.sh production mpi
```

On the UMass Unity cluster, submit the tested build and three-seed array with:

```bash
bash hpc/submit_unity.sh
source LAST_SPARTA_TUTORIAL_JOBS.env
squeue -j "${BUILD_JOB_ID},${ARRAY_JOB_ID}"
```

After the array finishes, combine the seeds with:

```bash
python3 scripts/ensemble_postprocess.py \
  runs/production_kn01_seed_20260803 \
  runs/production_kn01_seed_20260819 \
  runs/production_kn01_seed_20260831 \
  --output runs/production_kn01_ensemble
```

## Completed example results

The example results were produced by Slurm array job 62778322 using 16 MPI CPU
ranks for each of three independent seeds. All runs completed and produced the
final averaged grid at sampling step 26,000.

| Seed | Warm-up loop (s) | Sampling loop (s) | Sampling steps/s | Collisions | Stuck particles |
|---:|---:|---:|---:|---:|---:|
| 20260803 | 138.664 | 326.124 | 79.724 | 53,397,088 | 0 |
| 20260819 | 69.750 | 132.906 | 195.627 | 53,404,533 | 0 |
| 20260831 | 69.593 | 149.983 | 173.353 | 53,407,045 | 0 |

The three-seed ensemble gives:

| Quantity | Result |
|---|---:|
| Maximum ensemble-mean speed | 67.337 m/s |
| Ensemble-mean domain temperature range | 294.570-314.102 K |
| Mean lid-adjacent temperature | 311.036 K |
| 11-cell lid temperature range | 304.378-312.991 K |
| Mean lid cellwise seed standard deviation | 1.199 K |

The velocity field contains the expected primary recirculating vortex driven
by the translating lid. The largest cell-centered speed occurs in the thin
high-speed region adjacent to the moving wall. The temperature field is
nonuniform even though every wall is held at 300 K because molecular momentum
transfer, viscous heating, expansion, and nonequilibrium transport redistribute
energy inside the cavity.

The raw lid values and their 95% Student-t seed interval are retained in
`ensemble_lid_profile.csv`. The plotted heavy curve is an explicitly labelled
11-cell moving average used only to improve readability. Field figures use a
sigma-one-cell Gaussian display filter, while the numerical CSV output remains
unsmoothed.

## Reading the output

The most important generated files are:

- `case_metadata.json`, which records dimensional and numerical inputs;
- `log.cavity`, which reports particles, collisions, timing, and performance;
- `grid.final.00026000`, which contains time-averaged cell fields;
- `lid_profile_raw.csv`, which preserves the top-cell-center values;
- `run_summary.json`, which reports each seed without a pass/fail gate;
- `ensemble_lid_profile.csv`, which contains the three-seed mean and interval;
- `ensemble_lid_profiles.png` and `ensemble_fields.png`, which are ready for the
  tutorial chapter.

## Recommended classroom sequence

1. Run the unit tests and the smoke case.
2. Inspect `argon.vss` and verify that `alpha = 1.0` selects VHS.
3. Compare `smoke`, `tutorial`, and `production` metadata rather than treating
   a coarse run as quantitative evidence.
4. Read `log.cavity` and identify particle count, collision count, loop time,
   and parallel performance.
5. Plot the raw lid profile before using the labelled display average.
6. Repeat with a new random seed and use the ensemble script to quantify
   stochastic variability.

