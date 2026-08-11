#!/usr/bin/env bash
#SBATCH --job-name=cavity-models
#SBATCH --partition=gpu
#SBATCH --output=logs/cavity-%A_%a.out
#SBATCH --error=logs/cavity-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --constraint=2080ti
#SBATCH --array=0-4

set -euo pipefail

MODELS=(ntc-prescan sbt gbt ssbt sgbt)
MODEL="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
CONFIG_PATH="${1:-configs/unity_kn01_model_matrix.toml}"
OUTPUT_ROOT="${2:-results/unity_kn01_model_matrix}"
ENV_NAME="${DSMC_CAVITY_ENV:-dsmc-gpu}"
SEED="${DSMC_CAVITY_SEED:-20260803}"

mkdir -p "${OUTPUT_ROOT}/${MODEL}"
export PYTHONPATH="${PWD}/src"

mamba run -n "${ENV_NAME}" python -c \
  "import cupy as cp; print('CUDA devices:', cp.cuda.runtime.getDeviceCount(), 'runtime:', cp.cuda.runtime.runtimeGetVersion())"
mamba run -n "${ENV_NAME}" python scripts/run_aligned_model.py \
  --config "${CONFIG_PATH}" \
  --model "${MODEL}" \
  --backend gpu \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_ROOT}/${MODEL}" \
  --no-progress
