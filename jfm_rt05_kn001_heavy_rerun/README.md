# JFM RT=0.5, Kn=0.01 heavy endpoint reruns

This package reruns only the three noise-limited continuum endpoints identified
by the ensemble audit:

- HS at `R_T=0.5`, `Kn=0.01`
- BGK at `R_T=0.5`, `Kn=0.01`
- Shakhov at `R_T=0.5`, `Kn=0.01`

Each model is run with independent seeds `42`, `271828`, and `314159`, giving
nine independent GPU tasks. Each task uses 22,000,000 simulator particles,
2,000,000 steps, sampling from step 400,000 every two steps, and four temporal
blocks. No smoothing, filtering, interpolation of the two-dimensional fields,
or velocity projection is applied.

## One-line Unity submission

Run this from any Unity login-node directory:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/agent/jfm-kn001-heavy-reruns/jfm_rt05_kn001_heavy_rerun/submit_from_github.sh)
```

The bootstrap refreshes the code under
`/project/pi_roohie_umass_edu/JFM_revision_2026/JFM_RT05_KN001_HEAVY_RERUN`
without deleting existing outputs, runs preflight checks, submits the nine-task
GPU array with concurrency nine, and submits a dependent CPU summary job. No
smoke job is submitted.

To use a different output or installation path, export `JFM_OUTPUT_ROOT`,
`JFM_PROJECT_ROOT`, or `JFM_RERUN_ROOT` before running the one-line command.

## Slurm resources

- partition: `gpu` (non-preemptible)
- QoS: `long` (jobs longer than 48 hours)
- GPU: one `2080ti` per task
- array: `0-8%9`
- CPU/memory per GPU task: 4 CPUs, 64 GB
- time limit: 168 hours

## Output and quality decision

Raw run outputs are in `run_output/kn001_heavy`. After all nine tasks complete,
`run_output/summary_kn001_heavy` contains direct three-seed means, sample SD,
standard errors, profiles, and corrected kinetic-energy diagnostics.

The reported corrected estimator is

```text
(1/3) integral [rho_3 (u_1 dot u_2) + rho_2 (u_1 dot u_3)
              + rho_1 (u_2 dot u_3)] dA.
```

Every velocity cross-product is weighted by the density from the third
independent seed. The three pair values, their range, and a quality status are
written to both NPZ/JSON and `ALL_ENSEMBLES_summary.csv`.

Use the new endpoint quantitatively only if all pair estimates are positive and
their relative range is acceptably small. The summary labels pair-range <=10%
as `GOOD`, 10-25% as `USABLE_WITH_UNCERTAINTY`, larger spread as `PROVISIONAL`,
and a non-positive mean or pair as `NOISE_LIMITED`.

The high-Kn light points are not rerun here. Their existing uncertainty-aware
integral estimates and the high-Kn asymptotic solution are sufficient unless a
fully numerical precision endpoint is explicitly requested by the coauthors.
