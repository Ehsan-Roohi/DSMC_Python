# Unity OpenFOAM / dsmcFoam Runbook

Last verified: 2026-08-12 (America/New_York)

Purpose: persistent instructions for running OpenFOAM `dsmcFoam` on the UMass Unity cluster without repeating module-loading mistakes.

## Non-negotiable Unity rule

Do **not** run either of these on the Unity login node (`cpu001`):

```bash
module load uri/main
module load OpenFOAM/v2406-foss-2023a
```

The `uri/main` module branch is compiled for AVX-512 compute nodes and intentionally stops the shell on the login node. Its message is:

```text
NOTE: The modules under this branch will not run on the login node.
Use --constraint=avx512 for sbatch or srun sessions.
```

Consequently, `command -v dsmcFoam`, `module list`, or any other command placed after `module load uri/main` on the login node will never execute. This does **not** mean OpenFOAM is missing and does **not** justify installing another copy.

## Confirmed module hierarchy

Unity's module spider reports the required parent for OpenFOAM v2406 as:

```text
uri/main
```

The correct module sequence, executed **inside an AVX-512 Slurm allocation**, is:

```bash
module purge
module load uri/main
module load OpenFOAM/v2406-foss-2023a
command -v dsmcFoam
```

## Required Slurm directives

Every Unity job that uses this OpenFOAM module must contain at least:

```bash
#SBATCH --partition=cpu
#SBATCH --constraint=avx512
```

For the current lid-driven DSMC cavity case, the committed job uses:

```bash
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --constraint=avx512
#SBATCH --mem=64G
#SBATCH --time=48:00:00
```

Only after Slurm starts the job on an AVX-512 node should the script load `uri/main` and `OpenFOAM/v2406-foss-2023a`.

## Canonical repository and branch

- Repository: `Ehsan-Roohi/DSMC_Python`
- Branch: `agent/dsmcfoam-lid-driven-cavity`
- Package: `dsmcfoam_lid_driven_cavity`
- Slurm file: `dsmcfoam_lid_driven_cavity/hpc/unity_dsmcfoam_kn005.slurm`
- Launcher: `dsmcfoam_lid_driven_cavity/hpc/run_unity.sh`
- Dedicated Unity checkout: `/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK/DSMC_Python_dsmcFoam`

This is an OpenFOAM `dsmcFoam` case. It is not SPARTA and must not touch existing SPARTA cases or their untracked log files.

## Canonical submission command

Run this on the login node without loading any OpenFOAM module first:

```bash
bash /project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK/DSMC_Python_dsmcFoam/dsmcfoam_lid_driven_cavity/hpc/run_unity.sh
```

The launcher updates or creates a dedicated checkout, submits the job, and writes the current job information to:

```text
/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK/LAST_DSMCFOAM_KN005_JOB.env
```

## Status command

```bash
source /project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK/LAST_DSMCFOAM_KN005_JOB.env
sacct -X -j "$JOB_ID" --format=JobID%18,JobName%22,State,ExitCode,Elapsed,NodeList%20
tail -100 "$SLURM_LOG" 2>/dev/null || true
tail -100 "${SLURM_LOG%.out}.err" 2>/dev/null || true
```

## Safe module preflight

If module verification is needed, do it in Slurm, never directly on `cpu001`. Example:

```bash
srun --partition=cpu --constraint=avx512 --nodes=1 --ntasks=1 --time=00:05:00 --mem=2G \
  bash -lc 'module purge; module load uri/main; module load OpenFOAM/v2406-foss-2023a; command -v dsmcFoam; dsmcFoam -help | head'
```

## Mistakes that must not be repeated

1. Do not assume the module is absent because `dsmcFoam` is not visible on the login node.
2. Do not load `uri/main` on the login node; it terminates the remaining shell command sequence.
3. Do not guess parent modules such as `foss/2023a`. Use `module --show_hidden spider <module>` when the hierarchy is unknown.
4. Do not install a second OpenFOAM copy until the existing module has been tested inside a compatible Slurm allocation.
5. Do not paste Markdown links such as `[URL](URL)` into Bash. Commands must contain only the raw URL when a URL is unavoidable.
6. Do not use the SPARTA repository/branch for this case.
7. Do not submit the OpenFOAM module on a node lacking `avx512`.

## Current scientific baseline

- Solver: OpenFOAM `dsmcFoam`
- Gas: monatomic argon, VHS
- Cavity length: `1e-6 m`
- Top wall: `100 m/s`
- Wall temperature: `300 K`
- Knudsen number: `0.05`
- Mesh: `80 x 80 x 1`
- MPI ranks: `16`, decomposition `4 x 4 x 1`

Before any future Unity OpenFOAM/dsmcFoam task, retrieve and follow this runbook first.
