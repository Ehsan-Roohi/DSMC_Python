#!/usr/bin/env bash
#SBATCH --job-name=moh_mv1
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
REPO_ROOT="${MV1_REPO_ROOT:?exported by submit script}"
M3_ROOT="${MV1_M3_ROOT:?exported by submit script}"
OUTPUT_ROOT="${MV1_OUTPUT_ROOT:?exported by submit script}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv-mv1/bin/activate"
python -m vgdsmc.mohammadzadeh_vision \
  --m3-root "${M3_ROOT}" \
  --output-dir "${OUTPUT_ROOT}" \
  --epochs 160 --batch-size 6
python -m vgdsmc.mohammadzadeh_vision \
  --output-dir "${OUTPUT_ROOT}" --verify-only \
  > "${OUTPUT_ROOT}/verification.json"
