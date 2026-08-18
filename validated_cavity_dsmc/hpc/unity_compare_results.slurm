#!/usr/bin/env bash
#SBATCH --job-name=cavity-compare
#SBATCH --partition=cpu
#SBATCH --output=logs/cavity-compare-%j.out
#SBATCH --error=logs/cavity-compare-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

OUTPUT_ROOT="${1:-results/unity_kn01_model_matrix}"
ENV_NAME="${DSMC_CAVITY_ENV:-dsmc-gpu}"
MPL_DIR="$(mktemp -d "/tmp/dsmc-mpl-${SLURM_JOB_ID}-XXXXXX")"
trap 'rm -rf "${MPL_DIR}"' EXIT
export MPLCONFIGDIR="${MPL_DIR}"
export PYTHONPATH="${PWD}/src"

mamba run -n "${ENV_NAME}" python scripts/compare_completed_runs.py \
  --root "${OUTPUT_ROOT}" \
  --models ntc-prescan sbt gbt ssbt sgbt
