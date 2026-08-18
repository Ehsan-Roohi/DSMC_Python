#!/usr/bin/env bash
#SBATCH --job-name=m3_qy_r100
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --array=0-7%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

REPO_ROOT="${M3_REPO_ROOT:?M3_REPO_ROOT must be exported by the submit script}"
OUTPUT_ROOT="${M3_OUTPUT_ROOT:?M3_OUTPUT_ROOT must be exported by the submit script}"
SEEDS=(91901 91902 91903 91904 91905 91906 91907 91908)
SEED="${SEEDS[${SLURM_ARRAY_TASK_ID}]}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv-m3/bin/activate"
python -m vgdsmc.mohammadzadeh_qy_precision \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_ROOT}/seed_${SEED}"
