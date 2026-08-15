#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${MV16B_PROJECT_ROOT:-/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc}"
PAYLOAD_ROOT="${PROJECT_ROOT}/mv16b_jcp_evidence_audit"
MV15C_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV15C_A1_QY_RESULT.env"
MV15C_JOB_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV15C_A1_QY_JOB.env"
MV11_POINTER="${PROJECT_ROOT}/LAST_MV11_DS2V_CYLINDER_RESULT.env"
MV16A_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV16A_CYLINDER_JOB.env"
POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV16B_JCP_EVIDENCE_JOB.env"
RESULT_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV16B_JCP_EVIDENCE_RESULT.env"

command -v sbatch >/dev/null
for required in "${MV15C_POINTER}" "${MV15C_JOB_POINTER}" "${MV11_POINTER}" "${MV16A_POINTER}"; do
  test -s "${required}"
done
source "${MV15C_POINTER}"
source "${MV15C_JOB_POINTER}"
source "${MV11_POINTER}"
source "${MV16A_POINTER}"
: "${MV15C_A1_OUTPUT_ROOT:?}"
: "${MV11_CAMPAIGN_ROOT:?}"
: "${MV16A_OUTPUT_ROOT:?}"
: "${MV16A_PREDICT_JOB_ID:?}"

PYTHON_BIN="${MV16B_PYTHON:-${MV15C_A1_PYTHON:-}}"
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "MV16B_SUBMIT_ERROR: usable MV16B/MV15C-A1 Python not found" >&2
  exit 2
fi
if [[ -f "${POINTER}" && "${MV16B_ALLOW_RESUBMIT:-0}" != "1" ]]; then
  echo "MV16B_SUBMIT_ERROR: refusing duplicate submission; inspect ${POINTER}" >&2
  exit 3
fi
if [[ -e "${RESULT_POINTER}" ]]; then
  echo "MV16B_SUBMIT_ERROR: result pointer already exists; refusing overwrite" >&2
  exit 4
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${MV16B_OUTPUT_ROOT:-${PROJECT_ROOT}/results/mohammadzadeh_2012/mv16b_jcp_evidence_audit/run_${STAMP}}"
test ! -e "${OUTPUT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

cd "${PROJECT_ROOT}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
"${PYTHON_BIN}" -m vgdsmc.mohammadzadeh_mv16b_jcp_evidence_audit verify

EXPORTS="ALL,MV16B_PROJECT_ROOT=${PROJECT_ROOT},MV16B_PAYLOAD_ROOT=${PAYLOAD_ROOT},MV16B_OUTPUT_ROOT=${OUTPUT_ROOT},MV16B_PYTHON=${PYTHON_BIN},MV16B_MV15C_ROOT=${MV15C_A1_OUTPUT_ROOT},MV16B_MV16A_ROOT=${MV16A_OUTPUT_ROOT},MV16B_CAMPAIGN_ROOT=${MV11_CAMPAIGN_ROOT},MV16B_BATCH_SIZE=${MV16B_BATCH_SIZE:-4}"
AUDIT_JOB="$(sbatch --parsable --dependency="afterok:${MV16A_PREDICT_JOB_ID}" \
  --export="${EXPORTS}" \
  --output="${PROJECT_ROOT}/logs/moh_mv16b_audit_%j.slurm.out" \
  --error="${PROJECT_ROOT}/logs/moh_mv16b_audit_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv16b_audit.sbatch")"
AUDIT_JOB="${AUDIT_JOB%%;*}"
POST_JOB="$(sbatch --parsable --dependency="afterok:${AUDIT_JOB}" \
  --export="${EXPORTS},MV16B_AUDIT_JOB_ID=${AUDIT_JOB}" \
  --output="${PROJECT_ROOT}/logs/moh_mv16b_post_%j.slurm.out" \
  --error="${PROJECT_ROOT}/logs/moh_mv16b_post_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv16b_post.sbatch")"
POST_JOB="${POST_JOB%%;*}"

TEMPORARY="${POINTER}.tmp"
cat > "${TEMPORARY}" <<EOF
MV16B_OUTPUT_ROOT=${OUTPUT_ROOT}
MV16B_PYTHON=${PYTHON_BIN}
MV16B_MV15C_ROOT=${MV15C_A1_OUTPUT_ROOT}
MV16B_MV16A_ROOT=${MV16A_OUTPUT_ROOT}
MV16B_CAMPAIGN_ROOT=${MV11_CAMPAIGN_ROOT}
MV16B_PARENT_MV16A_PREDICT_JOB_ID=${MV16A_PREDICT_JOB_ID}
MV16B_AUDIT_JOB_ID=${AUDIT_JOB}
MV16B_POST_JOB_ID=${POST_JOB}
MV16B_JOB_IDS=${AUDIT_JOB},${POST_JOB}
EOF
mv "${TEMPORARY}" "${POINTER}"
echo "MV16B_JCP_EVIDENCE_SUBMITTED output=${OUTPUT_ROOT} audit=${AUDIT_JOB} post=${POST_JOB} dependency=afterok:${MV16A_PREDICT_JOB_ID} dsmc_rerun=NO training=NO"

