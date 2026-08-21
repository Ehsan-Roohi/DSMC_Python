#!/bin/bash
#SBATCH --job-name=j6r-repair
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -Eeuo pipefail
: "${JCP6R_CODE:?}"
: "${JCP6R_WORK:?}"
: "${JCP6R_JCP4:?}"
: "${JCP6R_M10_ROOT:?}"
: "${JCP6R_FAILED_JCP6:?}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
python "${JCP6R_CODE}/scripts/jcp6r_repair_freeze.py" \
  --jcp4 "${JCP6R_JCP4}" \
  --m10-root "${JCP6R_M10_ROOT}" \
  --failed-jcp6 "${JCP6R_FAILED_JCP6}" \
  --protocol "${JCP6R_CODE}/reference_data/mohammadzadeh_2012/jcp6r_repair_protocol.json" \
  --output "${JCP6R_WORK}"
cd "${JCP6R_WORK}"
sha256sum -c JCP6R_MODEL_LOCK.zip.sha256
echo "JCP6R_MODEL_LOCK_COMPLETE=1"
echo "UPLOAD_ZIP=${JCP6R_WORK}/JCP6R_MODEL_LOCK.zip"
echo "UPLOAD_SHA=${JCP6R_WORK}/JCP6R_MODEL_LOCK.zip.sha256"
