#!/usr/bin/env bash
#SBATCH --job-name=moh_mv10_qy
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --array=0-2%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${MV10_REPO_ROOT:?}"
source "${MV10_VENV_DIR:?}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
python -m vgdsmc.mohammadzadeh_mv10_qy_multiscale \
  --mode model \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --output-root "${MV10_OUTPUT_ROOT:?}" \
  --epochs "${MV10_EPOCHS:-240}" \
  --batch-size "${MV10_BATCH_SIZE:-8}"
