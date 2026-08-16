# SPARTA Kn=0.1 high-statistics rerun

This workflow reduces Monte Carlo noise by collecting more raw statistics. It
does not smooth, filter, or alter the reported SPARTA fields.

The fifth field in every new grid dump is the translational thermal temperature
from SPARTA's `compute thermal/grid` command.  It subtracts the cell's
center-of-mass velocity before converting random kinetic energy to temperature.
Do not use `compute grid ... temp` for this moving-lid case: that diagnostic
includes bulk streaming kinetic energy and therefore overstates temperature,
especially next to the moving wall.

The original HQ job 62962579 used the non-thermal diagnostic.  Reconstructing
its thermal temperature as
`T_thermal = T_grid - m*(u^2+v^2+w^2)/(3*kB)` reduces the full-field temperature
RMSE against the dsmcFoam HQ ensemble from 0.975 K to 0.156 K.  The rerun below
records the correct diagnostic directly.

## Statistical configuration

| Setting | Earlier production | HQ rerun |
|---|---:|---:|
| Grid | 200 x 200 | 200 x 200 |
| Independent seeds | 3 | 3 |
| Initial particles/cell | 32 | 128 |
| Warmup steps | 14,000 | 40,000 |
| Averaging steps | 26,000 | 160,000 |
| Sampling stride | 10 | 10 |

The sampling particle-step budget is 24.6 times larger than the earlier
production ensemble. If the remaining scatter is dominated by independent
Monte Carlo noise, the expected standard-deviation reduction is about
`sqrt(24.6) = 4.96`.

The Knudsen number, cavity size, gas model, number density, wall temperature,
lid speed, grid, and timestep remain unchanged. Each campaign uses the explicit
seeds `20260803`, `20260819`, and `20260831`.

## Submit on Unity

Run this on a Unity login node:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/agent/sparta-kn01-hq/sparta_cavity_mohammadzadeh/hpc/bootstrap_unity_sparta_kn01_hq.sh)
```

The command uses a dedicated checkout at
`/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK/DSMC_Python_sparta_kn01_hq`, runs
the unit tests, builds the pinned MPI SPARTA executable, executes a short
`thermal/grid` smoke case, submits a three-member CPU array only after all of
those checks pass, and submits a collector after the array.

Monitor all jobs with:

```bash
ROOT=/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK; source "$ROOT/LAST_SPARTA_KN01_HQ_JOBS.env"; squeue -j "$HQ_BUILD_JOB_ID,$HQ_ARRAY_JOB_ID,$HQ_COLLECT_JOB_ID"; sacct -X -j "$HQ_BUILD_JOB_ID,$HQ_ARRAY_JOB_ID,$HQ_COLLECT_JOB_ID" --format=JobID%22,JobName%20,State,ExitCode,Elapsed,NodeList%22
```

After the collector completes, upload the path stored in `HQ_RETURN_BUNDLE`.
The archive includes the three generated input decks, exact metadata, raw
averaged grid dumps, SPARTA logs, and Slurm logs.
