#!/usr/bin/env bash
#SBATCH --job-name=j2-predict
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

: "${JCP2_REPO_ROOT:?}"
: "${JCP2_RUN_ROOT:?}"
: "${JCP2_PREDICTION_ROOT:?}"
: "${JCP2_MV9_ROOT:?}"
: "${JCP2_MV15C_ROOT:?}"
: "${JCP2_PYTHON:?}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${JCP2_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${JCP2_REPO_ROOT}"
"${JCP2_PYTHON}" -m vgdsmc.jcp_phase1_cavity predict \
  --run-root "${JCP2_RUN_ROOT}" \
  --mv9-root "${JCP2_MV9_ROOT}" \
  --mv15c-root "${JCP2_MV15C_ROOT}" \
  --output-dir "${JCP2_PREDICTION_ROOT}" \
  --batch-size 8

test -s "${JCP2_PREDICTION_ROOT}/predictions.npz"
test -s "${JCP2_PREDICTION_ROOT}/prediction_lock.json"

