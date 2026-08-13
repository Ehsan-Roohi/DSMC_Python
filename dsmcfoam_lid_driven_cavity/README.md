# OpenFOAM dsmcFoam: lid-driven micro-cavity

This is an OpenFOAM `dsmcFoam` case, not a SPARTA case.

Baseline:

- monatomic argon, VHS (`mass=6.63e-26 kg`, `diameter=4.17e-10 m`, `omega=0.81`)
- square cavity `L=1e-6 m`; periodic depth is one collision cell (`1.25e-8 m`)
- top wall velocity `(100 0 0) m/s`; all walls at `300 K`
- fully diffuse Maxwellian thermal walls
- `Kn=0.05`, using `lambda=1/(sqrt(2)*pi*d^2*n)`
- number density `2.588767306032e25 1/m3`
- `80 x 80 x 1` cells and approximately 32 simulator particles per cell
- `deltaT=5e-12 s`, 20,000 steps; averaging begins at `2e-8 s`
- 16 MPI ranks (`4 x 4 x 1` decomposition)

The dictionaries follow the OpenFOAM `dsmcFoam` tutorial syntax. The Unity job script first uses an already-loaded OpenFOAM environment, then tries Unity's hierarchical module stacks (`foss/2023a` + `OpenFOAM/v2406-foss-2023a`, followed by the 2020a stack), common module names, and installation paths. If your installation has a non-standard path, submit with `DSMCFOAM_BASHRC=/absolute/path/to/OpenFOAM/etc/bashrc`.

From Unity, the public bootstrap command clones this branch into a dedicated checkout and submits the Slurm job:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/agent/dsmcfoam-lid-driven-cavity/dsmcfoam_lid_driven_cavity/hpc/run_unity.sh)
```

The command records the job ID in:

```text
/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK/LAST_DSMCFOAM_KN005_JOB.env
```

Monitor with:

```bash
source /project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK/LAST_DSMCFOAM_KN005_JOB.env && squeue -j "$JOB_ID" && tail -f "$SLURM_LOG"
```
