# Unity GPU launch

From the package directory:

```bash
sbatch hpc/unity_gpu_validation.slurm configs/production_mohammadzadeh_kn01.toml
```

The script uses `mamba run`, not `mamba shell` or interactive activation. Set
`DSMC_CAVITY_ENV` if the CuPy environment has another name. Confirm the local
Unity partition/account policy before submission; the included resource lines
only encode the known 2080 Ti constraint.
