# SPARTA lid-driven cavity: Mohammadzadeh benchmark

This self-contained teaching case maps the rarefied lid-driven cavity of
Mohammadzadeh et al., *Physical Review E* 85, 056310 (2012), to SPARTA. It gives
students one-command case generation, serial/MPI execution, centerline
post-processing, and guarded Codex assistance.

> Evidence boundary: the smoke workflow has been executed successfully with
> official SPARTA source. It verifies syntax and the complete data path. The
> production comparison is deliberately labelled pending until the documented
> resolution, repeat-seed, and uncertainty study is complete.

## Benchmark specification

| Item | Value used here | Source/interpretation |
|---|---:|---|
| Gas | monatomic argon | paper |
| Molecular mass | 6.63e-26 kg | paper |
| Reference diameter | 4.17e-10 m | paper |
| VHS viscosity index | 0.81 | companion model used in this repository |
| Wall/lid temperature | 300 K | paper |
| Lid velocity | 100 m/s | paper |
| Wall interaction | fully diffuse, full thermal accommodation | paper |
| Primary case | Kn = 0.1 | paper also reports 0.05 and 0.005 |
| Production grid | 200 x 200 | paper's selected grid |
| Initial population | 32 simulator particles/cell | paper |
| Collision selection | VHS with no-time-counter selection | paper/SPARTA |

The paper defines the problem through Knudsen number but does not fix an
absolute cavity length. This implementation chooses `L = 1e-6 m` and preserves
the similarity parameters. With `lambda = Kn L`, the number density is

```text
n = 1 / (sqrt(2) pi d_ref^2 lambda).
```

For SPARTA's two-dimensional unit-depth formulation, the particle weight is

```text
fnum = n L^2 / N_sim.
```

The production Kn = 0.1 case has `Delta x/lambda = 0.05`. The smoke and student
presets are learning runs and are not resolution evidence.

## 1. Download from GitHub

```bash
git clone --branch agent/validated-dsmc-cavity --single-branch \
  https://github.com/Ehsan-Roohi/DSMC_Python.git
cd DSMC_Python/sparta_cavity_mohammadzadeh
```

On Ubuntu/Debian, install the prerequisites once:

```bash
sudo apt update
sudo apt install -y git build-essential openmpi-bin libopenmpi-dev \
  python3 python3-venv
```

The SPARTA installer is intentionally local: it clones the official source at a
pinned commit and never writes into system directories.

## 2. Build SPARTA

Serial build:

```bash
bash scripts/install_sparta_linux.sh serial
```

MPI build:

```bash
bash scripts/install_sparta_linux.sh mpi
```

The official SPARTA documentation also supports CMake and documents `-in` and
`-log` command-line switches: <https://sparta.github.io/doc/Section_start.html>.

## 3. Test and run

```bash
python3 -m unittest discover -s tests -v
bash scripts/run_case.sh smoke serial
bash scripts/run_case.sh student serial
```

The smoke run is very short. A noisy or failed reference metric is expected; its
purpose is to catch parser, boundary, collision, averaging, and post-processing
errors. The student preset is suitable for learning and plotting, not a
publication claim.

Outputs are written under `runs/<level>_kn01/`:

- `in.cavity` and `case_metadata.json`: complete reproducibility record;
- `log.cavity`: SPARTA run log;
- `grid.final.*`: averaged grid fields;
- `lid_profile.csv`: top-cell velocity-slip and temperature profile;
- `validation_metrics.json` and `validation.png`: reference comparison.

Install optional plotting dependencies in a virtual environment if needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 4. MPI and production runs

```bash
MPI_RANKS=8 bash scripts/run_case.sh student mpi
MPI_RANKS=16 bash scripts/run_case.sh production mpi
```

Production is a CPU/MPI case. On the UMass Unity cluster, this one-line command
clones or updates the book branch, submits a CPU build of pinned MPI SPARTA,
submits three independent production seeds as a dependent CPU job array, and
submits a final collector job:

```bash
bash <(curl -fsSL \
  https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/agent/validated-dsmc-cavity/sparta_cavity_mohammadzadeh/hpc/bootstrap_unity_sparta_kn01_production.sh)
```

The production matrix uses a `200 x 200` grid, 32 initial simulator particles
per cell, 16 MPI ranks, and seeds `20260803`, `20260819`, and `20260831`. It runs
only on the `cpu` partition and asks for no GPU. The bootstrap prints the job
IDs, monitoring commands, and the exact `SPARTA_KN01_RESULTS_<job>.tar.gz` file
to return for ensemble post-processing and confidence intervals. Existing seed
directories are never overwritten.

GPU SPARTA requires a separate Kokkos/CUDA-enabled build. This repository does
not silently request a GPU or describe a CPU binary as GPU-enabled.

## 5. One-command bootstrap

From an empty working directory with the prerequisites already installed:

```bash
bash <(curl -fsSL \
  https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/agent/validated-dsmc-cavity/sparta_cavity_mohammadzadeh/scripts/bootstrap_linux.sh)
```

This clones the branch, builds serial SPARTA, runs the unit test, and completes
the smoke workflow. It refuses to overwrite an existing destination.

## 6. Run SPARTA with Codex

Codex can inspect the case, run tests and SPARTA, and explain logs while remaining
inside this working tree. Install the CLI using the official OpenAI command:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

Interactive use:

```bash
cd DSMC_Python/sparta_cavity_mohammadzadeh
codex
```

Ask it to read `AGENTS.md` and execute the guarded smoke workflow. For a
reproducible non-interactive run, first build serial SPARTA and then use:

```bash
bash scripts/run_with_codex.sh
```

The wrapper uses `codex exec --sandbox workspace-write`, saves the final report
to `runs/codex_smoke_report.md`, forbids `sudo`, and does not start production
without an explicit request. See the official
[Codex CLI documentation](https://developers.openai.com/codex/cli/) and
[CLI reference](https://developers.openai.com/codex/cli/reference/).

## 7. Reproducible case generation

Canonical generated decks are committed under `cases/`. Regenerate a case with:

```bash
python3 scripts/generate_case.py --level production --kn 0.1 \
  --seed 20260803 --output runs/production_kn01
```

Do not edit generated dimensional values by hand. Change the generator, record
the change, and rerun the tests. The surface model follows SPARTA's documented
moving diffuse wall syntax (`translate Ux Uy Uz`):
<https://sparta.github.io/doc/surf_collide.html>.

## Validation policy

Read [`VALIDATION_STATUS.md`](VALIDATION_STATUS.md) before presenting results.
The digitized paper data are retained in `reference/`; raw and filtered
comparisons must be reported separately.
