#!/usr/bin/env bash
#SBATCH --job-name=moh_mv5_assemble
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${MV5R_REPO_ROOT:?exported by submit script}"
ORIGINAL_ROOT="${MV5R_ORIGINAL_ROOT:?exported by submit script}"
ASSEMBLED_ROOT="${MV5R_ASSEMBLED_ROOT:?exported by submit script}"
VENV_DIR="${MV5R_VENV_DIR:?exported by submit script}"

cd "${REPO_ROOT}"
source "${VENV_DIR}/bin/activate"
python -m vgdsmc.mohammadzadeh_mv5_reference_stability_repair \
  --mode assemble \
  --output-root "${ORIGINAL_ROOT}" \
  --assembled-root "${ASSEMBLED_ROOT}"
