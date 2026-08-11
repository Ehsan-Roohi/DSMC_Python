# Unity GPU launch

From the package directory:

```bash
sbatch hpc/unity_gpu_validation.slurm configs/production_mohammadzadeh_kn01.toml
```

The script uses `mamba run`, not `mamba shell` or interactive activation. Set
`DSMC_CAVITY_ENV` if the CuPy environment has another name. Confirm the local
Unity partition/account policy before submission; the included resource lines
only encode the known 2080 Ti constraint.

## Five-model matrix from a fresh Unity login

The bootstrap script clones or safely updates the GitHub branch, submits five
independent GPU array tasks (`ntc-prescan`, `sbt`, `gbt`, `ssbt`, `sgbt`), and
then submits a dependent CPU comparison job.  From the Unity home directory:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/agent/validated-dsmc-cavity/validated_cavity_dsmc/hpc/bootstrap_unity_kn01_matrix.sh)
```

The default is the already CPU-validated 50x50, Kn=0.1 matrix.  The runner
aligns every model to the same physical warmup, sampling interval, and end
time; a Bernoulli model therefore receives more steps whenever its probability
limit selects a smaller `dt`.  After all array tasks succeed, the comparison
job writes `comparison.csv`, `comparison.md`, and `comparison.png` under
`results/unity_kn01_model_matrix`.

To launch the 200x200 production configuration later, override the config and
output root for the same one-line bootstrap:

```bash
DSMC_CAVITY_CONFIG=configs/production_mohammadzadeh_kn01.toml \
DSMC_CAVITY_OUTPUT=results/production_kn01_model_matrix \
bash <(curl -fsSL https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/agent/validated-dsmc-cavity/validated_cavity_dsmc/hpc/bootstrap_unity_kn01_matrix.sh)
```
