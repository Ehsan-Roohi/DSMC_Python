#!/usr/bin/env bash
set -Eeuo pipefail

JCP7_CLOSE_CODE_COMMIT=086f1eea4ee03291e8b101f80536929b7fb68a57
JCP7_RAW="https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/${JCP7_CLOSE_CODE_COMMIT}"
JCP7_WORK=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP7_M12_EVALUATION
JCP7_CODE="${JCP7_WORK}/code"
JCP7_MODEL_LOCK=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP6R_MODEL_LOCK/JCP6R_MODEL_LOCK.zip
EXPECTED_MODEL_LOCK_SHA256=bcb57b4585f9be949c8c859cf2d5036a1570499794cf402599f162119390fd20

trap 'RC=$?; echo "JCP7_CLOSE_BOOTSTRAP_FAILED rc=${RC} line=${LINENO} command=${BASH_COMMAND}" >&2; exit "${RC}"' ERR

if [[ -f "${JCP7_WORK}/JCP7_M12_EVALUATION.zip" && -f "${JCP7_WORK}/JCP7_M12_EVALUATION.zip.sha256" && \
      -f "${JCP7_WORK}/JCP7_M12_PREDICTION_LOCK.zip" && -f "${JCP7_WORK}/JCP7_M12_PREDICTION_LOCK.zip.sha256" ]]; then
  if (cd "${JCP7_WORK}" && sha256sum -c JCP7_M12_EVALUATION.zip.sha256 JCP7_M12_PREDICTION_LOCK.zip.sha256); then
    echo "JCP7_ALREADY_CLOSED=1"
    echo "UPLOAD=${JCP7_WORK}/JCP7_M12_EVALUATION.zip ${JCP7_WORK}/JCP7_M12_EVALUATION.zip.sha256 ${JCP7_WORK}/JCP7_M12_PREDICTION_LOCK.zip ${JCP7_WORK}/JCP7_M12_PREDICTION_LOCK.zip.sha256"
    exit 0
  fi
fi

[[ -d "${JCP7_WORK}/units" ]] || { echo "MISSING_JCP7_UNITS=1" >&2; exit 2; }
[[ -f "${JCP7_MODEL_LOCK}" ]] || { echo "MISSING_JCP6R_MODEL_LOCK=1" >&2; exit 2; }
[[ "$(sha256sum "${JCP7_MODEL_LOCK}" | awk '{print $1}')" == "${EXPECTED_MODEL_LOCK_SHA256}" ]] || { echo "MODEL_LOCK_CHECKSUM_MISMATCH=1" >&2; exit 2; }

mkdir -p "${JCP7_CODE}/scripts" "${JCP7_CODE}/reference_data/mohammadzadeh_2012" "${JCP7_WORK}/logs"
FILES=(
  scripts/jcp6_train_freeze.py
  scripts/jcp6r_repair_freeze.py
  scripts/jcp7_lock_m12_predictions.py
  scripts/jcp7_close_existing_m12.py
  scripts/unity_jcp7_close_existing_m12.sbatch
  reference_data/mohammadzadeh_2012/jcp7_m12_closeout_amendment.json
)
for FILE in "${FILES[@]}"; do
  mkdir -p "${JCP7_CODE}/$(dirname "${FILE}")"
  curl --retry 3 -fsSL "${JCP7_RAW}/${FILE}" -o "${JCP7_CODE}/${FILE}"
done
python -m py_compile "${JCP7_CODE}/scripts/"*.py
bash -n "${JCP7_CODE}/scripts/unity_jcp7_close_existing_m12.sbatch"

JCP7_CLOSE_JOB_ID="$(sbatch --parsable \
  --output="${JCP7_WORK}/logs/j7-close_%j.out" \
  --error="${JCP7_WORK}/logs/j7-close_%j.err" \
  --export="ALL,JCP7_WORK=${JCP7_WORK},JCP7_CODE=${JCP7_CODE},JCP7_MODEL_LOCK=${JCP7_MODEL_LOCK}" \
  "${JCP7_CODE}/scripts/unity_jcp7_close_existing_m12.sbatch")"
printf 'JCP7_CLOSE_JOB_ID=%q\nJCP7_WORK=%q\nJCP7_CLOSE_CODE_COMMIT=%q\n' \
  "${JCP7_CLOSE_JOB_ID}" "${JCP7_WORK}" "${JCP7_CLOSE_CODE_COMMIT}" > "${JCP7_WORK}/LAST_JCP7_CLOSE.env"
echo "JCP7_CLOSE_EXISTING_SUBMITTED=1"
echo "JCP7_CLOSE_JOB_ID=${JCP7_CLOSE_JOB_ID}"
echo "MONITOR=squeue -j ${JCP7_CLOSE_JOB_ID}"
echo "WHEN_COMPLETE_UPLOAD=${JCP7_WORK}/JCP7_M12_EVALUATION.zip ${JCP7_WORK}/JCP7_M12_EVALUATION.zip.sha256 ${JCP7_WORK}/JCP7_M12_PREDICTION_LOCK.zip ${JCP7_WORK}/JCP7_M12_PREDICTION_LOCK.zip.sha256"
