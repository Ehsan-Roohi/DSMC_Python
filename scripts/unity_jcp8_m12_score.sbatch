#!/bin/bash
#SBATCH --job-name=j8-m12-score
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
set -Eeuo pipefail
: "${JCP8_WORK:?}"; : "${JCP8_CODE:?}"; : "${JCP8_PREDICTION_LOCK:?}"; : "${JCP8_MODEL_LOCK:?}"
python "${JCP8_CODE}/scripts/score_jcp8_m12.py" --prediction "${JCP8_PREDICTION_LOCK}" --reference "${JCP8_WORK}/JCP8_M12_REFERENCE.zip" --model-lock "${JCP8_MODEL_LOCK}" --protocol "${JCP8_CODE}/reference_data/mohammadzadeh_2012/jcp8_m12_reference_score_protocol.json" --output "${JCP8_WORK}"
cd "${JCP8_WORK}"; sha256sum -c JCP8_M12_SCORE.zip.sha256
echo "JCP8_SCORE_COMPLETE=1"
echo "UPLOAD=${JCP8_WORK}/JCP8_M12_REFERENCE.zip ${JCP8_WORK}/JCP8_M12_REFERENCE.zip.sha256 ${JCP8_WORK}/JCP8_M12_SCORE.zip ${JCP8_WORK}/JCP8_M12_SCORE.zip.sha256"
