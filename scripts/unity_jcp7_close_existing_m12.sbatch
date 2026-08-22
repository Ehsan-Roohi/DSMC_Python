#!/bin/bash
#SBATCH --job-name=j7-close
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00

set -Eeuo pipefail
: "${JCP7_WORK:?}"
: "${JCP7_CODE:?}"
: "${JCP7_MODEL_LOCK:?}"

python "${JCP7_CODE}/scripts/jcp7_close_existing_m12.py" \
  --work "${JCP7_WORK}" \
  --model-lock "${JCP7_MODEL_LOCK}" \
  --amendment "${JCP7_CODE}/reference_data/mohammadzadeh_2012/jcp7_m12_closeout_amendment.json"

cd "${JCP7_WORK}"
sha256sum -c JCP7_M12_EVALUATION.zip.sha256 JCP7_M12_PREDICTION_LOCK.zip.sha256
echo "JCP7_CLOSE_EXISTING_COMPLETE=1"
echo "UPLOAD=${JCP7_WORK}/JCP7_M12_EVALUATION.zip ${JCP7_WORK}/JCP7_M12_EVALUATION.zip.sha256 ${JCP7_WORK}/JCP7_M12_PREDICTION_LOCK.zip ${JCP7_WORK}/JCP7_M12_PREDICTION_LOCK.zip.sha256"
