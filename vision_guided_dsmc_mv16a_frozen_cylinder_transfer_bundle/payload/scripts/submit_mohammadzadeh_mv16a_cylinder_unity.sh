#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${MV16A_PROJECT_ROOT:-/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc}"
PAYLOAD_ROOT="${PROJECT_ROOT}/mv16a_frozen_cylinder_transfer"
MV11_RESULT_POINTER="${PROJECT_ROOT}/LAST_MV11_DS2V_CYLINDER_RESULT.env"
MV15C_A1_RESULT_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV15C_A1_QY_RESULT.env"
MV15C_A1_JOB_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV15C_A1_QY_JOB.env"
POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV16A_CYLINDER_JOB.env"
RESULT_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV16A_CYLINDER_RESULT.env"

command -v sbatch >/dev/null
test -d "${PROJECT_ROOT}"
test -d "${PAYLOAD_ROOT}"
test -s "${MV11_RESULT_POINTER}"
test -s "${MV15C_A1_RESULT_POINTER}"
test -s "${MV15C_A1_JOB_POINTER}"
source "${MV11_RESULT_POINTER}"
source "${MV15C_A1_RESULT_POINTER}"
source "${MV15C_A1_JOB_POINTER}"
: "${MV11_CAMPAIGN_ROOT:?}"
: "${MV15C_A1_OUTPUT_ROOT:?}"

PYTHON_BIN="${MV16A_PYTHON:-${MV15C_A1_PYTHON:-}}"
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "MV16A_SUBMIT_ERROR: MV16A_PYTHON/MV15C_A1_PYTHON is unavailable" >&2
  exit 2
fi
if [[ -f "${POINTER}" && "${MV16A_ALLOW_RESUBMIT:-0}" != "1" ]]; then
  echo "MV16A_SUBMIT_ERROR: refusing duplicate submission; inspect ${POINTER}" >&2
  exit 3
fi
if [[ -e "${RESULT_POINTER}" ]]; then
  echo "MV16A_SUBMIT_ERROR: result pointer already exists; refusing overwrite" >&2
  exit 4
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${MV16A_OUTPUT_ROOT:-${PROJECT_ROOT}/results/mohammadzadeh_2012/mv16a_frozen_cylinder_transfer/run_${STAMP}}"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

cd "${PROJECT_ROOT}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
"${PYTHON_BIN}" -m vgdsmc.mohammadzadeh_mv16a_frozen_cylinder_transfer verify
"${PYTHON_BIN}" -m vgdsmc.mohammadzadeh_mv16a_frozen_cylinder_transfer prepare \
  --campaign-root "${MV11_CAMPAIGN_ROOT}" \
  --mv15c-output-root "${MV15C_A1_OUTPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}"

EXPORTS="ALL,MV16A_PROJECT_ROOT=${PROJECT_ROOT},MV16A_PAYLOAD_ROOT=${PAYLOAD_ROOT},MV16A_OUTPUT_ROOT=${OUTPUT_ROOT},MV16A_PYTHON=${PYTHON_BIN},MV16A_BATCH_SIZE=${MV16A_BATCH_SIZE:-4}"
PREDICT_JOB="$(sbatch --parsable --export="${EXPORTS}" \
  --output="${PROJECT_ROOT}/logs/moh_mv16a_cylinder_predict_%j.slurm.out" \
  --error="${PROJECT_ROOT}/logs/moh_mv16a_cylinder_predict_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv16a_cylinder_predict.sbatch")"
PREDICT_JOB="${PREDICT_JOB%%;*}"
POST_JOB="$(sbatch --parsable --dependency="afterok:${PREDICT_JOB}" \
  --export="${EXPORTS},MV16A_PREDICT_JOB_ID=${PREDICT_JOB}" \
  --output="${PROJECT_ROOT}/logs/moh_mv16a_cylinder_post_%j.slurm.out" \
  --error="${PROJECT_ROOT}/logs/moh_mv16a_cylinder_post_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv16a_cylinder_post.sbatch")"
POST_JOB="${POST_JOB%%;*}"

TEMPORARY="${POINTER}.tmp"
cat > "${TEMPORARY}" <<EOF
MV16A_OUTPUT_ROOT=${OUTPUT_ROOT}
MV16A_PYTHON=${PYTHON_BIN}
MV16A_MV11_CAMPAIGN_ROOT=${MV11_CAMPAIGN_ROOT}
MV16A_MV15C_A1_OUTPUT_ROOT=${MV15C_A1_OUTPUT_ROOT}
MV16A_PREDICT_JOB_ID=${PREDICT_JOB}
MV16A_POST_JOB_ID=${POST_JOB}
MV16A_JOB_IDS=${PREDICT_JOB},${POST_JOB}
EOF
mv "${TEMPORARY}" "${POINTER}"
echo "MV16A_CYLINDER_SUBMITTED output=${OUTPUT_ROOT} predict=${PREDICT_JOB} post=${POST_JOB} dsmc_rerun=NO training=NO"
