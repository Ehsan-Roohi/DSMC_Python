#!/usr/bin/env bash
set -Eeuo pipefail

JCP8_CODE_COMMIT=fa752170561e820f115b079390f9ed63e3382e57
JCP8_RAW="https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/${JCP8_CODE_COMMIT}"
JCP8_SOURCE_DIR=/project/pi_roohie_umass_edu/Ab-initio-shock/ABINITIO_SHOCK_TESTS_v2/DS2V_BIRD_M10_FRESH_ONLY_20260722_234904/source
JCP8_DATA_ROOT=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc/mv11_ds2v_cylinder_runs
JCP8_PREDICTION_LOCK=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP7_M12_EVALUATION/JCP7_M12_PREDICTION_LOCK.zip
JCP8_MODEL_LOCK=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP6R_MODEL_LOCK/JCP6R_MODEL_LOCK.zip
JCP8_WORK=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP8_M12_REFERENCE
JCP8_CODE="${JCP8_WORK}/code"
EXPECTED_DATA_SHA256=a13e82650ffa7a0303b0353ad385b198839c2c738df7cff98ce343806e736b96
EXPECTED_HEAT_BENCH_SHA256=2d94da3d86786afd1c497994cad935cfca1d188d9431bf16960fbc533e3f6c34
EXPECTED_PREDICTION_SHA256=54db6c0be71764df87f9912090821d4676625ea7ccd8da1f4c069e7edd2ac0d8
EXPECTED_MODEL_SHA256=bcb57b4585f9be949c8c859cf2d5036a1570499794cf402599f162119390fd20

trap 'RC=$?; echo "JCP8_BOOTSTRAP_FAILED rc=${RC} line=${LINENO} command=${BASH_COMMAND}" >&2; exit "${RC}"' ERR

if [[ -f "${JCP8_WORK}/JCP8_M12_REFERENCE.zip" && -f "${JCP8_WORK}/JCP8_M12_REFERENCE.zip.sha256" && -f "${JCP8_WORK}/JCP8_M12_SCORE.zip" && -f "${JCP8_WORK}/JCP8_M12_SCORE.zip.sha256" ]]; then
  if (cd "${JCP8_WORK}" && sha256sum -c JCP8_M12_REFERENCE.zip.sha256 JCP8_M12_SCORE.zip.sha256); then
    echo "JCP8_ALREADY_COMPLETE=1"
    echo "UPLOAD=${JCP8_WORK}/JCP8_M12_REFERENCE.zip ${JCP8_WORK}/JCP8_M12_REFERENCE.zip.sha256 ${JCP8_WORK}/JCP8_M12_SCORE.zip ${JCP8_WORK}/JCP8_M12_SCORE.zip.sha256"
    exit 0
  fi
fi

[[ -f "${JCP8_SOURCE_DIR}/Plasma_Calculations2.bird_m10_fresh.F90" ]] || { echo "MISSING_DS2V_SOURCE=1" >&2; exit 2; }
[[ -f "${JCP8_PREDICTION_LOCK}" ]] || { echo "MISSING_JCP7_PREDICTION_LOCK=1" >&2; exit 2; }
[[ "$(sha256sum "${JCP8_PREDICTION_LOCK}" | awk '{print $1}')" == "${EXPECTED_PREDICTION_SHA256}" ]] || { echo "PREDICTION_LOCK_CHECKSUM_MISMATCH=1" >&2; exit 2; }
[[ -f "${JCP8_MODEL_LOCK}" ]] || { echo "MISSING_JCP6R_MODEL_LOCK=1" >&2; exit 2; }
[[ "$(sha256sum "${JCP8_MODEL_LOCK}" | awk '{print $1}')" == "${EXPECTED_MODEL_SHA256}" ]] || { echo "MODEL_LOCK_CHECKSUM_MISMATCH=1" >&2; exit 2; }

JCP8_DATA=
while IFS= read -r CANDIDATE; do
  if [[ "$(sha256sum "${CANDIDATE}" | awk '{print $1}')" == "${EXPECTED_DATA_SHA256}" ]]; then JCP8_DATA="${CANDIDATE}"; break; fi
done < <(find "${JCP8_DATA_ROOT}" -type f -path '*/input/DS2VD.DAT' -print)
JCP8_HEAT_BENCH=
while IFS= read -r CANDIDATE; do
  if [[ "$(sha256sum "${CANDIDATE}" | awk '{print $1}')" == "${EXPECTED_HEAT_BENCH_SHA256}" ]]; then JCP8_HEAT_BENCH="${CANDIDATE}"; break; fi
done < <(find "${JCP8_DATA_ROOT}" -type f -path '*/input/HEAT-BENCH.TXT' -print)
[[ -n "${JCP8_DATA}" && -f "${JCP8_DATA}" ]] || { echo "MISSING_LOCKED_DS2V_DATA=1" >&2; exit 2; }
[[ -n "${JCP8_HEAT_BENCH}" && -f "${JCP8_HEAT_BENCH}" ]] || { echo "MISSING_LOCKED_HEAT_BENCH=1" >&2; exit 2; }

mkdir -p "${JCP8_CODE}/scripts" "${JCP8_CODE}/reference_data/mohammadzadeh_2012" "${JCP8_WORK}/logs" "${JCP8_WORK}/units" "${JCP8_WORK}/cases"
FILES=(
  scripts/prepare_jcp3_ds2v_m12.py
  scripts/prepare_jcp8_ds2v_m12.py
  scripts/jcp6_train_freeze.py
  scripts/verify_jcp8_m12_reference.py
  scripts/collect_jcp8_m12_reference.py
  scripts/score_jcp8_m12.py
  scripts/unity_jcp8_m12_reference.sbatch
  scripts/unity_jcp8_m12_collect.sbatch
  scripts/unity_jcp8_m12_score.sbatch
  reference_data/mohammadzadeh_2012/jcp8_m12_reference_score_protocol.json
)
for FILE in "${FILES[@]}"; do
  mkdir -p "${JCP8_CODE}/$(dirname "${FILE}")"
  curl --retry 3 -fsSL "${JCP8_RAW}/${FILE}" -o "${JCP8_CODE}/${FILE}"
done
python -m py_compile "${JCP8_CODE}/scripts/"*.py
bash -n "${JCP8_CODE}/scripts/unity_jcp8_m12_reference.sbatch" "${JCP8_CODE}/scripts/unity_jcp8_m12_collect.sbatch" "${JCP8_CODE}/scripts/unity_jcp8_m12_score.sbatch"

ARRAY_JOB_ID="$(sbatch --parsable --output="${JCP8_WORK}/logs/j8-ref_%A_%a.out" --error="${JCP8_WORK}/logs/j8-ref_%A_%a.err" --export="ALL,JCP8_SOURCE_DIR=${JCP8_SOURCE_DIR},JCP8_DATA=${JCP8_DATA},JCP8_HEAT_BENCH=${JCP8_HEAT_BENCH},JCP8_WORK=${JCP8_WORK},JCP8_CODE=${JCP8_CODE}" "${JCP8_CODE}/scripts/unity_jcp8_m12_reference.sbatch")"
COLLECT_JOB_ID="$(sbatch --parsable --dependency="afterok:${ARRAY_JOB_ID}" --output="${JCP8_WORK}/logs/j8-pack_%j.out" --error="${JCP8_WORK}/logs/j8-pack_%j.err" --export="ALL,JCP8_WORK=${JCP8_WORK},JCP8_CODE=${JCP8_CODE}" "${JCP8_CODE}/scripts/unity_jcp8_m12_collect.sbatch")"
SCORE_JOB_ID="$(sbatch --parsable --dependency="afterok:${COLLECT_JOB_ID}" --output="${JCP8_WORK}/logs/j8-score_%j.out" --error="${JCP8_WORK}/logs/j8-score_%j.err" --export="ALL,JCP8_WORK=${JCP8_WORK},JCP8_CODE=${JCP8_CODE},JCP8_PREDICTION_LOCK=${JCP8_PREDICTION_LOCK},JCP8_MODEL_LOCK=${JCP8_MODEL_LOCK}" "${JCP8_CODE}/scripts/unity_jcp8_m12_score.sbatch")"
printf 'JCP8_ARRAY_JOB_ID=%q\nJCP8_COLLECT_JOB_ID=%q\nJCP8_SCORE_JOB_ID=%q\nJCP8_WORK=%q\nJCP8_CODE_COMMIT=%q\n' "${ARRAY_JOB_ID}" "${COLLECT_JOB_ID}" "${SCORE_JOB_ID}" "${JCP8_WORK}" "${JCP8_CODE_COMMIT}" > "${JCP8_WORK}/LAST_JCP8.env"
echo "JCP8_SUBMITTED=1"
echo "JCP8_ARRAY_JOB_ID=${ARRAY_JOB_ID}"
echo "JCP8_COLLECT_JOB_ID=${COLLECT_JOB_ID}"
echo "JCP8_SCORE_JOB_ID=${SCORE_JOB_ID}"
echo "MONITOR=squeue -j ${ARRAY_JOB_ID},${COLLECT_JOB_ID},${SCORE_JOB_ID}"
echo "WHEN_COMPLETE_UPLOAD=${JCP8_WORK}/JCP8_M12_REFERENCE.zip ${JCP8_WORK}/JCP8_M12_REFERENCE.zip.sha256 ${JCP8_WORK}/JCP8_M12_SCORE.zip ${JCP8_WORK}/JCP8_M12_SCORE.zip.sha256"
