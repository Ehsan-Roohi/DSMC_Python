#!/usr/bin/env bash
#SBATCH --job-name=moh_mv2
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --array=0-31%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${MV2_REPO_ROOT:?exported by submit script}"
M3_ROOT="${MV2_M3_ROOT:?exported by submit script}"
OUTPUT_ROOT="${MV2_OUTPUT_ROOT:?exported by submit script}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv-mv2/bin/activate"
python -m vgdsmc.mohammadzadeh_vision_mv2 \
  --m3-root "${M3_ROOT}" \
  --output-dir "${OUTPUT_ROOT}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --epochs 160 \
  --batch-size 6
