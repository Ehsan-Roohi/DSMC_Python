# SPARTA tutorial: rarefied lid-driven cavity

This self-contained example shows how to generate, run, and post-process a
two-dimensional lid-driven cavity with SPARTA. The case is intended for
teaching the complete workflow rather than validating against an external
dataset.

The gas is monatomic argon, all walls are fully diffuse at 300 K, and the top
wall translates at 100 m/s. The main case has `Kn = 0.1`. Collisions use
SPARTA's `vss` style with `alpha = 1`, which is the VHS limit.

## Repository layout

```text
data/               argon species and VHS collision data
hpc/                Unity/Slurm build and production scripts
scripts/            case generator, runner, and post-processing tools
tests/              fast input-deck tests
results/            compact outputs from the completed three-seed example
```

Generated run directories are not overwritten. Raw DSMC data are retained;
any spatially smoothed curve is labelled as a display aid.

## Download the tutorial

```bash
git clone --branch agent/sparta-lid-driven-cavity-tutorial --single-branch \
  https://github.com/Ehsan-Roohi/DSMC_Python.git
cd DSMC_Python/sparta_lid_driven_cavity_tutorial
```

## 1. Requirements

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y git build-essential openmpi-bin libopenmpi-dev \
  python3 python3-venv
```

Optional plotting dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 2. Build SPARTA

The installer clones the official SPARTA source at a pinned commit and builds
inside `third_party/`; it does not install into a system directory.

```bash
bash scripts/install_sparta_linux.sh serial
bash scripts/install_sparta_linux.sh mpi
```

The resulting executables are:

```text
third_party/sparta/src/spa_serial
third_party/sparta/src/spa_mpi
```

## 3. Inspect the physical model

The production setup is:

| Quantity | Value |
|---|---:|
| Cavity length | `1.0e-6 m` |
| Knudsen number | `0.1` |
| Mean free path | `1.0e-7 m` |
| Grid | `200 x 200` |
| Cell size / mean free path | `0.05` |
| Simulator particles/cell | `32` |
| Initial simulator particles | `1,280,000` |
| Argon mass | `6.63e-26 kg` |
| VHS reference diameter | `4.17e-10 m` |
| VHS viscosity index | `0.81` |
| VHS reference temperature | `273 K` |
| VSS angular parameter | `alpha = 1.0` (VHS limit) |
| Wall temperature | `300 K` |
| Lid velocity | `100 m/s` |
| Warm-up | `14,000` steps |
| Sampling | `26,000` steps |
| Sampling stride | `10` steps |

Generate a deck without running SPARTA:

```bash
python3 scripts/generate_case.py \
  --level production --kn 0.1 --seed 20260803 \
  --output runs/production_kn01_seed_20260803
```

The generated `case_metadata.json` records every dimensional and numerical
parameter. The `in.cavity` file is ready for SPARTA.

## 4. Test, smoke, and tutorial runs

Run the fast checks first:

```bash
python3 -m unittest discover -s tests -v
bash scripts/run_case.sh smoke serial
```

Then run the classroom-sized case:

```bash
bash scripts/run_case.sh tutorial serial
```

MPI execution is selected with the second argument:

```bash
MPI_RANKS=8 bash scripts/run_case.sh tutorial mpi
```

The runner creates a seed-specific directory under `runs/`, executes SPARTA,
and calls the standalone post-processor to produce profiles, fields, and
runtime statistics.

## 5. Production run

```bash
SEED=20260803 MPI_RANKS=16 bash scripts/run_case.sh production mpi
```

To reproduce the three-seed example:

```bash
for SEED in 20260803 20260819 20260831; do
  SEED="$SEED" MPI_RANKS=16 bash scripts/run_case.sh production mpi
done

python3 scripts/ensemble_postprocess.py \
  runs/production_kn01_seed_20260803 \
  runs/production_kn01_seed_20260819 \
  runs/production_kn01_seed_20260831 \
  --output runs/production_kn01_ensemble
```

## 6. Unity/Slurm

From the tutorial root on Unity:

```bash
bash hpc/submit_unity.sh
source LAST_SPARTA_TUTORIAL_JOBS.env
```

Monitor the jobs with:

```bash
squeue --me
sacct -X -j "$BUILD_JOB_ID,$ARRAY_JOB_ID" \
  --format=JobID%24,JobName%22,State,ExitCode,Elapsed,MaxRSS,NodeList%20
```

After all three array tasks finish:

```bash
python3 scripts/ensemble_postprocess.py \
  runs/production_kn01_seed_20260803 \
  runs/production_kn01_seed_20260819 \
  runs/production_kn01_seed_20260831 \
  --output runs/production_kn01_ensemble
```

## 7. Output files

Each run contains:

- `in.cavity`: generated SPARTA input;
- `case_metadata.json`: complete setup record;
- `log.cavity`: SPARTA log;
- `grid.final.*`: time-averaged grid fields;
- `lid_profile_raw.csv`: raw top-cell-center profile;
- `lid_profile_11cell.csv`: labelled 11-cell display average;
- `run_summary.json`: numerical and runtime summary;
- `lid_profiles.png`: raw and display-smoothed lid profiles;
- `fields.png`: temperature and velocity field.

The ensemble directory additionally contains `ensemble_lid_profile.csv`,
`ensemble_summary.json`, `ensemble_lid_profiles.png`, and
`ensemble_fields.png`.

## 8. Completed example

The committed compact results were generated by SPARTA job array `62778322` on
16 CPU ranks for seeds `20260803`, `20260819`, and `20260831`. All three jobs
completed with zero stuck particles and produced the expected final dump at
step 26,000. The committed ensemble output contains the averaged field,
lid-adjacent profiles, seed uncertainty, and runtime information.
