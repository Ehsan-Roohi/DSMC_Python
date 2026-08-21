#!/usr/bin/env bash
set -Eeuo pipefail

JCP7_CODE_COMMIT=5bd541e2e222d884d1a3af3914f0edd4e5d6cdb5
JCP7_RAW="https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/${JCP7_CODE_COMMIT}"
JCP7_SOURCE_DIR=/project/pi_roohie_umass_edu/Ab-initio-shock/ABINITIO_SHOCK_TESTS_v2/DS2V_BIRD_M10_FRESH_ONLY_20260722_234904/source
JCP7_DATA_ROOT=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc/mv11_ds2v_cylinder_runs
JCP7_WORK=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP7_M12_EVALUATION
JCP7_CODE="${JCP7_WORK}/code"
JCP7_MODEL_LOCK=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP6R_MODEL_LOCK/JCP6R_MODEL_LOCK.zip
EXPECTED_DATA_SHA256=a13e82650ffa7a0303b0353ad385b198839c2c738df7cff98ce343806e736b96
EXPECTED_HEAT_BENCH_SHA256=2d94da3d86786afd1c497994cad935cfca1d188d9431bf16960fbc533e3f6c34
EXPECTED_MODEL_LOCK_SHA256=bcb57b4585f9be949c8c859cf2d5036a1570499794cf402599f162119390fd20

trap 'RC=$?; echo "JCP7_BOOTSTRAP_FAILED rc=${RC} line=${LINENO} command=${BASH_COMMAND}" >&2; exit "${RC}"' ERR

if [[ -f "${JCP7_WORK}/JCP7_M12_PREDICTION_LOCK.zip" && -f "${JCP7_WORK}/JCP7_M12_PREDICTION_LOCK.zip.sha256" ]]; then
  (cd "${JCP7_WORK}" && sha256sum -c JCP7_M12_PREDICTION_LOCK.zip.sha256)
  echo "JCP7_M12_PREDICTION_ALREADY_LOCKED=1"
  echo "UPLOAD=${JCP7_WORK}/JCP7_M12_EVALUATION.zip ${JCP7_WORK}/JCP7_M12_EVALUATION.zip.sha256 ${JCP7_WORK}/JCP7_M12_PREDICTION_LOCK.zip ${JCP7_WORK}/JCP7_M12_PREDICTION_LOCK.zip.sha256"
  exit 0
fi
[[ -f "${JCP7_SOURCE_DIR}/Plasma_Calculations2.bird_m10_fresh.F90" ]] || { echo "MISSING_DS2V_SOURCE=1" >&2; exit 2; }
[[ "$(sha256sum "${JCP7_MODEL_LOCK}" | awk '{print $1}')" == "${EXPECTED_MODEL_LOCK_SHA256}" ]] || { echo "MODEL_LOCK_CHECKSUM_MISMATCH=1" >&2; exit 2; }
JCP7_DATA=
while IFS= read -r CANDIDATE; do
  if [[ "$(sha256sum "${CANDIDATE}" | awk '{print $1}')" == "${EXPECTED_DATA_SHA256}" ]]; then
    JCP7_DATA="${CANDIDATE}"
    break
  fi
done < <(find "${JCP7_DATA_ROOT}" -type f -path '*/input/DS2VD.DAT' -print)
JCP7_HEAT_BENCH=
while IFS= read -r CANDIDATE; do
  if [[ "$(sha256sum "${CANDIDATE}" | awk '{print $1}')" == "${EXPECTED_HEAT_BENCH_SHA256}" ]]; then
    JCP7_HEAT_BENCH="${CANDIDATE}"
    break
  fi
done < <(find "${JCP7_DATA_ROOT}" -type f -path '*/input/HEAT-BENCH.TXT' -print)
[[ -n "${JCP7_DATA}" && -f "${JCP7_DATA}" ]] || { echo "MISSING_LOCKED_DS2V_DATA=1" >&2; exit 2; }
[[ -n "${JCP7_HEAT_BENCH}" && -f "${JCP7_HEAT_BENCH}" ]] || { echo "MISSING_LOCKED_HEAT_BENCH=1" >&2; exit 2; }
[[ "$(sha256sum "${JCP7_DATA}" | awk '{print $1}')" == "${EXPECTED_DATA_SHA256}" ]] || { echo "DATA_CHECKSUM_MISMATCH=1" >&2; exit 2; }
[[ "$(sha256sum "${JCP7_HEAT_BENCH}" | awk '{print $1}')" == "${EXPECTED_HEAT_BENCH_SHA256}" ]] || { echo "HEAT_BENCH_CHECKSUM_MISMATCH=1" >&2; exit 2; }

mkdir -p "${JCP7_CODE}/scripts" "${JCP7_CODE}/reference_data/mohammadzadeh_2012" "${JCP7_WORK}/logs" "${JCP7_WORK}/units" "${JCP7_WORK}/cases"
FILES=(
  scripts/prepare_jcp3_ds2v_m12.py
  scripts/jcp6_train_freeze.py
  scripts/jcp6r_repair_freeze.py
  scripts/verify_jcp7_m12_evaluation.py
  scripts/collect_jcp7_m12_evaluation.py
  scripts/jcp7_lock_m12_predictions.py
  scripts/unity_jcp7_m12_evaluation.sbatch
  scripts/unity_jcp7_m12_collect.sbatch
  scripts/unity_jcp7_m12_predict.sbatch
  reference_data/mohammadzadeh_2012/jcp7_m12_evaluation_protocol.json
)
for FILE in "${FILES[@]}"; do
  mkdir -p "${JCP7_CODE}/$(dirname "${FILE}")"
  curl --retry 3 -fsSL "${JCP7_RAW}/${FILE}" -o "${JCP7_CODE}/${FILE}"
done
python -m py_compile "${JCP7_CODE}/scripts/"*.py
bash -n "${JCP7_CODE}/scripts/unity_jcp7_m12_evaluation.sbatch" "${JCP7_CODE}/scripts/unity_jcp7_m12_collect.sbatch" "${JCP7_CODE}/scripts/unity_jcp7_m12_predict.sbatch"

ARRAY_JOB_ID="$(sbatch --parsable --output="${JCP7_WORK}/logs/j7-eval_%A_%a.out" --error="${JCP7_WORK}/logs/j7-eval_%A_%a.err" \
  --export="ALL,JCP7_SOURCE_DIR=${JCP7_SOURCE_DIR},JCP7_DATA=${JCP7_DATA},JCP7_HEAT_BENCH=${JCP7_HEAT_BENCH},JCP7_WORK=${JCP7_WORK},JCP7_CODE=${JCP7_CODE}" \
  "${JCP7_CODE}/scripts/unity_jcp7_m12_evaluation.sbatch")"
COLLECT_JOB_ID="$(sbatch --parsable --dependency="afterany:${ARRAY_JOB_ID}" --output="${JCP7_WORK}/logs/j7-pack_%j.out" --error="${JCP7_WORK}/logs/j7-pack_%j.err" \
  --export="ALL,JCP7_WORK=${JCP7_WORK},JCP7_CODE=${JCP7_CODE}" "${JCP7_CODE}/scripts/unity_jcp7_m12_collect.sbatch")"
PREDICT_JOB_ID="$(sbatch --parsable --dependency="afterok:${COLLECT_JOB_ID}" --output="${JCP7_WORK}/logs/j7-pred_%j.out" --error="${JCP7_WORK}/logs/j7-pred_%j.err" \
  --export="ALL,JCP7_WORK=${JCP7_WORK},JCP7_CODE=${JCP7_CODE},JCP7_MODEL_LOCK=${JCP7_MODEL_LOCK}" "${JCP7_CODE}/scripts/unity_jcp7_m12_predict.sbatch")"
printf 'JCP7_ARRAY_JOB_ID=%q\nJCP7_COLLECT_JOB_ID=%q\nJCP7_PREDICT_JOB_ID=%q\nJCP7_WORK=%q\nJCP7_CODE_COMMIT=%q\n' \
  "${ARRAY_JOB_ID}" "${COLLECT_JOB_ID}" "${PREDICT_JOB_ID}" "${JCP7_WORK}" "${JCP7_CODE_COMMIT}" > "${JCP7_WORK}/LAST_JCP7.env"
echo "JCP7_M12_EVALUATION_SUBMITTED=1"
echo "JCP7_ARRAY_JOB_ID=${ARRAY_JOB_ID}"
echo "JCP7_COLLECT_JOB_ID=${COLLECT_JOB_ID}"
echo "JCP7_PREDICT_JOB_ID=${PREDICT_JOB_ID}"
echo "MONITOR=squeue -j ${ARRAY_JOB_ID},${COLLECT_JOB_ID},${PREDICT_JOB_ID}"
echo "WHEN_COMPLETE_UPLOAD=${JCP7_WORK}/JCP7_M12_EVALUATION.zip ${JCP7_WORK}/JCP7_M12_EVALUATION.zip.sha256 ${JCP7_WORK}/JCP7_M12_PREDICTION_LOCK.zip ${JCP7_WORK}/JCP7_M12_PREDICTION_LOCK.zip.sha256"
