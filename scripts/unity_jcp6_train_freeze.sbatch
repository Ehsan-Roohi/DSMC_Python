#!/bin/bash
#SBATCH --job-name=j6-train-lock
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -Eeuo pipefail
: "${JCP6_CODE:?}"
: "${JCP6_WORK:?}"
: "${JCP6_JCP4:?}"
: "${JCP6_M10_ROOT:?}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
python "${JCP6_CODE}/scripts/jcp6_train_freeze.py" \
  --jcp4 "${JCP6_JCP4}" \
  --m10-root "${JCP6_M10_ROOT}" \
  --protocol "${JCP6_CODE}/reference_data/mohammadzadeh_2012/jcp6_model_lock_protocol.json" \
  --output "${JCP6_WORK}"
cd "${JCP6_WORK}"
sha256sum -c JCP6_MODEL_LOCK.zip.sha256
echo "JCP6_MODEL_LOCK_COMPLETE=1"
echo "UPLOAD_ZIP=${JCP6_WORK}/JCP6_MODEL_LOCK.zip"
echo "UPLOAD_SHA=${JCP6_WORK}/JCP6_MODEL_LOCK.zip.sha256"
