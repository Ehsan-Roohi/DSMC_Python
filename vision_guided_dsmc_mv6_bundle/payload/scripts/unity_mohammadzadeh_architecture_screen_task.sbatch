#!/usr/bin/env bash
#SBATCH --job-name=moh_mv6_arch
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --array=0-11%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${MV6_REPO_ROOT:?exported by submit script}"
M3_ROOT="${MV6_M3_ROOT:?exported by submit script}"
MV3_ROOT="${MV6_MV3_ROOT:?exported by submit script}"
MV5_REFERENCE_ROOT="${MV6_MV5_REFERENCE_ROOT:?exported by submit script}"
OUTPUT_ROOT="${MV6_OUTPUT_ROOT:?exported by submit script}"
VENV_DIR="${MV6_VENV_DIR:?exported by submit script}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"
source "${VENV_DIR}/bin/activate"
python -m vgdsmc.mohammadzadeh_architecture_screen \
  --mode task \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --existing-m3-root "${M3_ROOT}" \
  --mv3-root "${MV3_ROOT}" \
  --mv5-reference-root "${MV5_REFERENCE_ROOT}" \
  --output-dir "${OUTPUT_ROOT}" \
  --epochs "${MV6_EPOCHS:-200}" \
  --batch-size "${MV6_BATCH_SIZE:-6}"
