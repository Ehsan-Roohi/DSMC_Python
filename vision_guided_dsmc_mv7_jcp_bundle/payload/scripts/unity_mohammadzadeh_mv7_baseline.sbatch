#!/usr/bin/env bash
#SBATCH --job-name=moh_mv7_base
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --array=0-3%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${MV7_REPO_ROOT:?exported by submit script}"
M3_ROOT="${MV7_M3_ROOT:?exported by submit script}"
MV3_ROOT="${MV7_MV3_ROOT:?exported by submit script}"
REFERENCE_ROOT="${MV7_REFERENCE_ROOT:?exported by submit script}"
OUTPUT_ROOT="${MV7_OUTPUT_ROOT:?exported by submit script}"
VENV_DIR="${MV7_VENV_DIR:?exported by submit script}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"
source "${VENV_DIR}/bin/activate"
python -m vgdsmc.mohammadzadeh_mv7_jcp_budget_matrix \
  --mode baseline \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --existing-m3-root "${M3_ROOT}" \
  --mv3-root "${MV3_ROOT}" \
  --reference-root "${REFERENCE_ROOT}" \
  --output-root "${OUTPUT_ROOT}"
