#!/usr/bin/env bash
#SBATCH --job-name=dsmc-cavity
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=2080ti

set -euo pipefail

CONFIG_PATH="${1:-configs/production_mohammadzadeh_kn01.toml}"
ENV_NAME="${DSMC_CAVITY_ENV:-dsmc-gpu}"

# `mamba run` is used intentionally; it does not require shell activation.
# CuPy must already be installed in the cluster environment; compute nodes do
# not need outbound package downloads.
mamba run -n "${ENV_NAME}" python -c \
  "import cupy as cp; print(cp.cuda.runtime.getDeviceCount(), cp.cuda.runtime.runtimeGetVersion())"
mamba run -n "${ENV_NAME}" python -m pip install -e . --no-deps
mamba run -n "${ENV_NAME}" dsmc-cavity --config "${CONFIG_PATH}" --backend gpu
