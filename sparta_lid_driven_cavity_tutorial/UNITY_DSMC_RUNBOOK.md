# Unity DSMC Runbook: OpenFOAM `dsmcFoam` and SPARTA

Updated: 2026-08-12  
Scope: Unity Research Computing cluster; repositories under `Ehsan-Roohi/DSMC_Python`.

## Mandatory rule before every future run

Before submitting anything, explicitly identify the solver as **OpenFOAM `dsmcFoam`** or **SPARTA** and consult this runbook. Do not substitute one solver's build, module, case, launcher, branch, or validation rules for the other.

Every run must record and check:

1. solver and scientific case identity;
2. repository, branch, and commit;
3. executable path or source/build commit;
4. MPI implementation and ABI compatibility;
5. Slurm resources, constraints, and dependencies;
6. preflight result before production submission;
7. expected output contract and final-step marker;
8. `sacct` state and exit code for every build, array, and collector job.

Keep OpenFOAM and SPARTA in separate checkouts and unique run directories. Never clean, reset, switch, or overwrite a working tree containing untracked logs/results. Never use `git add -A` on a run directory.

---

## A. OpenFOAM `dsmcFoam` on Unity

The detailed OpenFOAM reference is the Library file `Unity_OpenFOAM_dsmcFoam_Runbook.md`.

### Non-negotiable Unity module rule

`OpenFOAM/v2406-foss-2023a` is under the `uri/main` hierarchy. That hierarchy is intended for compute nodes and reports that jobs must request AVX-512. Loading it on the login node can terminate the shell command sequence before diagnostics or submission run.

Therefore:

- do not use `module load uri/main` as a login-node preflight;
- load `uri/main` and `OpenFOAM/v2406-foss-2023a` inside the Slurm script;
- include `#SBATCH --constraint=avx512`;
- check `command -v dsmcFoam` inside the allocated compute job;
- do not assume `module load OpenFOAM/...` alone is sufficient.

Canonical login-node submission:

```bash
bash /project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK/DSMC_Python_dsmcFoam/dsmcfoam_lid_driven_cavity/hpc/run_unity.sh
```

The job script, not the login shell, owns module loading and solver discovery. If `dsmcFoam` is absent in the compute job, inspect the module environment there or supply an explicit `DSMCFOAM_BASHRC`; do not silently run SPARTA or another OpenFOAM solver.

---

## B. SPARTA on Unity

### Canonical tutorial workflow

- Repository: `Ehsan-Roohi/DSMC_Python`
- Branch: `agent/sparta-lid-driven-cavity-tutorial`
- Package: `sparta_lid_driven_cavity_tutorial`
- Submit from a dedicated checkout with:

```bash
bash sparta_lid_driven_cavity_tutorial/hpc/submit_unity.sh
```

The maintained helper submits a short build/preflight job followed by a three-seed production array. Each task is independent and uses 16 CPU MPI ranks. It tests MPI, sends the SPARTA input deck on standard input, verifies the final grid dump, and then runs postprocessing. The runner must refuse to overwrite an existing run directory.

After submission:

```bash
source LAST_SPARTA_TUTORIAL_JOBS.env
squeue -j "${BUILD_JOB_ID},${ARRAY_JOB_ID}"
sacct -X -j "${BUILD_JOB_ID},${ARRAY_JOB_ID}" \
  --format=JobID%24,JobName%22,State,ExitCode,Elapsed,MaxRSS,NodeList%20
```

For a workflow that also defines `COLLECT_JOB_ID`:

```bash
source LAST_SPARTA_KN01_JOBS.env
sacct -X -j "${BUILD_JOB_ID},${ARRAY_JOB_ID},${COLLECT_JOB_ID}" \
  --format=JobID%24,JobName%22,State,ExitCode,Elapsed,MaxRSS,NodeList%20
```

Also inspect every array task's stdout/stderr and verify the expected final timestep/output. A collector finishing is not proof that simulations succeeded.

### Known good validation evidence

Tutorial array job `62778322` completed all three tasks for seeds `20260803`, `20260819`, and `20260831`. Each produced `grid.final.00026000` and reported zero stuck particles. This is evidence for the maintained tutorial workflow, not blanket validation of unrelated cases or binaries.

The production tutorial uses:

- Kn = 0.1, `L = 1e-6 m`, argon at 300 K;
- VHS parameters `m = 6.63e-26 kg`, `dref = 4.17e-10 m`, `omega = 0.81`, `Tref = 273 K`;
- full diffuse walls and a top wall translating at 100 m/s;
- `200 x 200 x 1` cells and nominal 32 particles per cell;
- `dt = 8.2567881869e-13 s`;
- 14,000 warmup steps, 26,000 sampled steps, sampling stride 10.

Do not call a smoke/student run “validated.” Publication claims require seed, grid, particles-per-cell, timestep, warmup/sample sensitivity, and uncertainty studies.

### Build, executable, and MPI rules

1. Record the SPARTA executable path and the exact source commit. The earlier validated package pinned official SPARTA commit `912c9e163c38ea5c3562d039e65215f6e2a4f3f8`.
2. Standard SPARTA and locally modified SPARTA are not interchangeable. Some prior binaries added surface pressure/slip/friction outputs; reproduce those only with the documented modified source.
3. Never mix MPI implementations. A `spa_mpi` linked against a Conda environment's `libmpi.so.40` must be launched with that environment's matching `mpirun`, not a module OpenMPI launcher.
4. Preserve `HOME`. Do not use `--export=NONE`; it previously caused OpenMPI failures such as “Unable to get the user home directory.”
5. Use `--nodes=1` for the maintained single-node CPU workflow. Earlier attempts without the expected node layout failed.
6. Do not use `srun --gres=none`; it produced an invalid-GRES error.
7. Set a writable job-local `TMPDIR` and, if needed, `OMPI_MCA_orte_tmpdir_base` when MPI temporary-directory problems occur.
8. Standard CPU SPARTA runs request no GPU/GRES. A GPU run needs a separately verified Kokkos/CUDA build; loading a GPU allocation does not convert a CPU binary into a GPU binary.

### Dependency and failure interpretation

- `(Dependency)` means a downstream job is waiting on an upstream condition; it is not a priority diagnosis.
- A collector that requires successful simulation outputs must use `afterok`, not `afterany`.
- Always judge build, each array task, and collector independently.
- Observed failures include array exit `213:0` with collector exit `6:0`, and a later array exit `4:0` after about three seconds while build and pack completed. The available records do not establish the root causes. Preserve and inspect task logs before changing code or resources.
- A completed pack job after failed array tasks is a workflow defect or non-success dependency choice; it must never be reported as a successful simulation.

Failure inspection template:

```bash
source LAST_SPARTA_KN01_JOBS.env
sacct -X -j "${BUILD_JOB_ID},${ARRAY_JOB_ID},${COLLECT_JOB_ID}" \
  --format=JobID%24,JobName%22,State,ExitCode,Elapsed,MaxRSS,NodeList%20

# Then inspect the exact stdout/stderr paths recorded by the submission helper.
# Do not resubmit until the first failing stage and its first meaningful error are known.
```

### Scientific invariants

- SPARTA's command is named `collide vss`, but `alpha = 1` gives the VHS angular law. Do not label an `alpha = 1.4` sensitivity case as the VHS baseline.
- Monatomic argon has no rotational degrees of freedom.
- The moving lid is represented by `surf_collide ... translate 100 0 0`, not by moving the freestream.
- In a two-dimensional unit-depth case, compute particle weight using `fnum = n L^2 / N_sim`, not `n L^3 / N_sim`.
- Generate and record derived quantities such as mean free path, number density, `fnum`, timestep, and `dx/lambda`; do not hand-edit them independently.
- Keep warmup and sampling intervals separate.
- Retain raw profiles. Any smoothing is display-only and must be labeled.
- Standard SPARTA uses Bird's NTC method, not NTC-PreScan. Do not claim NTC-PreScan unless the source actually implements it.
- Low particles per cell can weaken the initial `(sigmaT*gr)max` estimate in standard NTC; address this through documented sensitivity or source changes, not by renaming the method.
- Direct wall samples and top cell-center macroscopic samples are different observables.

### Output contract and completion test

A production task is successful only if all of the following agree:

1. Slurm state is `COMPLETED` with exit code `0:0`;
2. SPARTA log reaches the requested final timestep without a fatal error;
3. the expected final dump exists (tutorial: `grid.final.00026000`);
4. required metadata, profiles, and validation metrics exist;
5. stuck-particle and other physical sanity checks pass;
6. postprocessing reads the preserved raw task outputs.

Expected retained artifacts include `case_metadata.json`, `log.cavity`, the final grid dump, the lid profile, and validation metrics. Combine independent seeds only with the ensemble postprocessor; never replace or overwrite seed-level outputs.

Parallel output requires special care: multiple MPI ranks writing the same non-parallel-safe Tecplot file can corrupt it. The presence of a file does not prove completion—verify its timestep/final marker and use per-rank or SPARTA-supported parallel output.

### Performance rule

More MPI ranks are not automatically faster. A previous very large run spent roughly 60% in “Other” and 30% in communication with large Move/Coll imbalance. Before increasing ranks, inspect SPARTA timer fractions, per-rank variance, particles/cell, and load balance. If testing `fix balance`/RCB, treat it as a measured experiment and retain the baseline.

---

## C. Final pre-submit checklist for either solver

```text
[ ] Solver explicitly identified: dsmcFoam or SPARTA
[ ] Correct dedicated checkout, branch, and commit recorded
[ ] Existing untracked logs/results preserved
[ ] Executable found inside the compute environment
[ ] MPI launcher matches the executable's MPI library
[ ] Required Slurm constraint/resources included
[ ] Preflight passed before production
[ ] Unique run directory selected; overwrite refused
[ ] Dependencies use afterok when downstream output is required
[ ] Output contract and final-step marker written down
[ ] All build/array/collector jobs will be checked separately
```

If any item is unknown, stop before submitting and diagnose it. Never switch solvers merely because the requested executable is missing.
