#!/usr/bin/env bash
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

: "${JCP2_REPO_ROOT:?}"
: "${JCP2_RUN_ROOT:?}"
: "${JCP2_PYTHON:?}"
: "${JCP2_GROUP:?}"
: "${SLURM_ARRAY_TASK_ID:?}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${JCP2_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${JCP2_REPO_ROOT}"
"${JCP2_PYTHON}" -m vgdsmc.jcp_phase1_recovery recover-checkpoint \
  --group "${JCP2_GROUP}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --run-root "${JCP2_RUN_ROOT}"
