#!/usr/bin/env bash
set -Eeuo pipefail

JCP4_CODE_COMMIT=6f6171108af0ba00edc1dd310d7b892059026f7a
JCP4_RAW="https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/${JCP4_CODE_COMMIT}"
JCP4_SOURCE_DIR=/project/pi_roohie_umass_edu/Ab-initio-shock/ABINITIO_SHOCK_TESTS_v2/DS2V_BIRD_M10_FRESH_ONLY_20260722_234904/source
JCP4_DATA_SEARCH_ROOT=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc/mv11_ds2v_cylinder_runs
JCP4_EXPECTED_DATA_SHA256=a13e82650ffa7a0303b0353ad385b198839c2c738df7cff98ce343806e736b96
JCP4_EXPECTED_HEAT_BENCH_SHA256=2d94da3d86786afd1c497994cad935cfca1d188d9431bf16960fbc533e3f6c34
JCP4_WORK=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP4_M8_REFERENCE
JCP4_CODE="${JCP4_WORK}/code"

trap 'RC=$?; echo "JCP4_BOOTSTRAP_FAILED rc=${RC} line=${LINENO} command=${BASH_COMMAND}" >&2; exit "${RC}"' ERR

if [[ -f "${JCP4_WORK}/JCP4_M8_REFERENCE.zip" && -f "${JCP4_WORK}/JCP4_M8_REFERENCE.zip.sha256" ]]; then
  (cd "${JCP4_WORK}" && sha256sum -c JCP4_M8_REFERENCE.zip.sha256)
  echo "JCP4_M8_REFERENCE_ALREADY_COMPLETE=1"
  echo "UPLOAD=${JCP4_WORK}/JCP4_M8_REFERENCE.zip ${JCP4_WORK}/JCP4_M8_REFERENCE.zip.sha256"
  exit 0
fi

[[ -f "${JCP4_SOURCE_DIR}/Plasma_Calculations2.bird_m10_fresh.F90" ]] || { echo "MISSING_DS2V_SOURCE=1" >&2; exit 2; }

JCP4_DATA=
for CANDIDATE in \
  "${JCP4_SOURCE_DIR}/DS2VD.DAT" \
  "${JCP4_SOURCE_DIR}/../DS2VD.DAT" \
  "${JCP4_DATA_SEARCH_ROOT}/MV11_DS2V_CYLINDER_20260813_170355/input/DS2VD.DAT" \
  "${JCP4_DATA_SEARCH_ROOT}/MV11_DS2V_CYLINDER_20260813_133511/input/DS2VD.DAT"
do
  if [[ -f "${CANDIDATE}" ]]; then JCP4_DATA="${CANDIDATE}"; break; fi
done
if [[ -z "${JCP4_DATA}" && -d "${JCP4_DATA_SEARCH_ROOT}" ]]; then
  JCP4_DATA="$(find "${JCP4_DATA_SEARCH_ROOT}" -type f -path '*/input/DS2VD.DAT' -print -quit 2>/dev/null || true)"
fi
[[ -n "${JCP4_DATA}" && -f "${JCP4_DATA}" ]] || { echo "MISSING_DS2V_DATA=1" >&2; exit 2; }
JCP4_DATA_SHA256="$(sha256sum "${JCP4_DATA}" | awk '{print $1}')"
[[ "${JCP4_DATA_SHA256}" == "${JCP4_EXPECTED_DATA_SHA256}" ]] || { echo "DS2V_DATA_CHECKSUM_MISMATCH actual=${JCP4_DATA_SHA256}" >&2; exit 2; }

JCP4_HEAT_BENCH=
for CANDIDATE in \
  "$(dirname "${JCP4_DATA}")/HEAT-BENCH.TXT" \
  "${JCP4_DATA_SEARCH_ROOT}/MV11_DS2V_CYLINDER_20260813_170355/input/HEAT-BENCH.TXT" \
  "${JCP4_DATA_SEARCH_ROOT}/MV11_DS2V_CYLINDER_20260813_133511/input/HEAT-BENCH.TXT"
do
  if [[ -f "${CANDIDATE}" ]]; then JCP4_HEAT_BENCH="${CANDIDATE}"; break; fi
done
if [[ -z "${JCP4_HEAT_BENCH}" && -d "${JCP4_DATA_SEARCH_ROOT}" ]]; then
  JCP4_HEAT_BENCH="$(find "${JCP4_DATA_SEARCH_ROOT}" -type f -path '*/input/HEAT-BENCH.TXT' -print -quit 2>/dev/null || true)"
fi
[[ -n "${JCP4_HEAT_BENCH}" && -f "${JCP4_HEAT_BENCH}" ]] || { echo "MISSING_HEAT_BENCH=1" >&2; exit 2; }
JCP4_HEAT_BENCH_SHA256="$(sha256sum "${JCP4_HEAT_BENCH}" | awk '{print $1}')"
[[ "${JCP4_HEAT_BENCH_SHA256}" == "${JCP4_EXPECTED_HEAT_BENCH_SHA256}" ]] || { echo "HEAT_BENCH_CHECKSUM_MISMATCH actual=${JCP4_HEAT_BENCH_SHA256}" >&2; exit 2; }

mkdir -p "${JCP4_CODE}/scripts" "${JCP4_CODE}/reference_data/mohammadzadeh_2012" "${JCP4_WORK}/logs" "${JCP4_WORK}/units" "${JCP4_WORK}/cases"
FILES=(
  scripts/prepare_jcp3_ds2v_m12.py
  scripts/prepare_jcp4_ds2v_m8.py
  scripts/verify_jcp4_m8_reference.py
  scripts/collect_jcp4_m8_reference.py
  scripts/unity_jcp4_m8_reference.sbatch
  scripts/unity_jcp4_m8_collect.sbatch
  reference_data/mohammadzadeh_2012/jcp4_m8_reference_protocol.json
)
for FILE in "${FILES[@]}"; do
  mkdir -p "${JCP4_CODE}/$(dirname "${FILE}")"
  curl --retry 3 -fsSL "${JCP4_RAW}/${FILE}" -o "${JCP4_CODE}/${FILE}"
done
python -m py_compile "${JCP4_CODE}/scripts/prepare_jcp3_ds2v_m12.py" "${JCP4_CODE}/scripts/prepare_jcp4_ds2v_m8.py" "${JCP4_CODE}/scripts/verify_jcp4_m8_reference.py" "${JCP4_CODE}/scripts/collect_jcp4_m8_reference.py"
bash -n "${JCP4_CODE}/scripts/unity_jcp4_m8_reference.sbatch" "${JCP4_CODE}/scripts/unity_jcp4_m8_collect.sbatch"

ARRAY_JOB_ID="$(sbatch --parsable \
  --output="${JCP4_WORK}/logs/j4-m8-ref_%A_%a.out" \
  --error="${JCP4_WORK}/logs/j4-m8-ref_%A_%a.err" \
  --export="ALL,JCP4_SOURCE_DIR=${JCP4_SOURCE_DIR},JCP4_DATA=${JCP4_DATA},JCP4_HEAT_BENCH=${JCP4_HEAT_BENCH},JCP4_WORK=${JCP4_WORK},JCP4_CODE=${JCP4_CODE}" \
  "${JCP4_CODE}/scripts/unity_jcp4_m8_reference.sbatch")"
COLLECT_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${ARRAY_JOB_ID}" \
  --output="${JCP4_WORK}/logs/j4-m8-pack_%j.out" \
  --error="${JCP4_WORK}/logs/j4-m8-pack_%j.err" \
  --export="ALL,JCP4_WORK=${JCP4_WORK},JCP4_CODE=${JCP4_CODE}" \
  "${JCP4_CODE}/scripts/unity_jcp4_m8_collect.sbatch")"

printf 'JCP4_ARRAY_JOB_ID=%q\nJCP4_COLLECT_JOB_ID=%q\nJCP4_WORK=%q\nJCP4_CODE_COMMIT=%q\nJCP4_DATA=%q\nJCP4_DATA_SHA256=%q\nJCP4_HEAT_BENCH=%q\nJCP4_HEAT_BENCH_SHA256=%q\n' \
  "${ARRAY_JOB_ID}" "${COLLECT_JOB_ID}" "${JCP4_WORK}" "${JCP4_CODE_COMMIT}" \
  "${JCP4_DATA}" "${JCP4_DATA_SHA256}" "${JCP4_HEAT_BENCH}" "${JCP4_HEAT_BENCH_SHA256}" \
  > "${JCP4_WORK}/LAST_JCP4.env"
echo "JCP4_M8_REFERENCE_SUBMITTED=1"
echo "JCP4_ARRAY_JOB_ID=${ARRAY_JOB_ID}"
echo "JCP4_COLLECT_JOB_ID=${COLLECT_JOB_ID}"
echo "MONITOR=squeue -j ${ARRAY_JOB_ID},${COLLECT_JOB_ID}"
echo "WHEN_COMPLETE_UPLOAD=${JCP4_WORK}/JCP4_M8_REFERENCE.zip ${JCP4_WORK}/JCP4_M8_REFERENCE.zip.sha256"
