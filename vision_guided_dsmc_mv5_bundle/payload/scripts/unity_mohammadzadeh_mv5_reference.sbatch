#!/usr/bin/env bash
#SBATCH --job-name=moh_mv5_ref
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --array=0-15%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${MV5_REPO_ROOT:?exported by submit script}"
OUTPUT_ROOT="${MV5_OUTPUT_ROOT:?exported by submit script}"
VENV_DIR="${MV5_VENV_DIR:?exported by submit script}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"
source "${VENV_DIR}/bin/activate"
python -m vgdsmc.mohammadzadeh_mv5_reference \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --output-root "${OUTPUT_ROOT}"
