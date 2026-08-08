#!/usr/bin/env bash
#SBATCH --job-name=moh_mv3_stab
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${MV3_REPO_ROOT:?exported by submit script}"
OUTPUT_ROOT="${MV3_OUTPUT_ROOT:?exported by submit script}"
VENV_DIR="${MV3_VENV_DIR:?exported by submit script}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"
source "${VENV_DIR}/bin/activate"
python -m vgdsmc.mohammadzadeh_mv3_reference_stability_repair \
  --output-root "${OUTPUT_ROOT}"
