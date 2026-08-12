#!/usr/bin/env bash
#SBATCH --job-name=moh_mv8_assemble
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${MV8_REPO_ROOT:?}"
source "${MV8_VENV_DIR:?}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
python -m vgdsmc.mohammadzadeh_mv8_kinetic_moments \
  --mode assemble \
  --existing-m3-root "${MV8_M3_ROOT:?}" \
  --mv3-root "${MV8_MV3_ROOT:?}" \
  --reference-root "${MV8_REFERENCE_ROOT:?}" \
  --output-root "${MV8_OUTPUT_ROOT:?}"
