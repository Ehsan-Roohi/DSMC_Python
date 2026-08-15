#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${MV15C_A1_PROJECT_ROOT:-/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc}"
PAYLOAD_ROOT="${PROJECT_ROOT}/mv15c_a1_qy_evaluation"
ORIGINAL_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV15C_FRESH_B3_JOB.env"
POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV15C_A1_QY_JOB.env"
command -v sbatch >/dev/null
test -d "${PROJECT_ROOT}"
test -d "${PAYLOAD_ROOT}"
test -s "${ORIGINAL_POINTER}"
source "${ORIGINAL_POINTER}"
: "${MV15C_OUTPUT_ROOT:?}"
: "${MV15C_REFERENCE_JOB_ID:?}"
: "${MV15C_PREDICT_JOB_ID:?}"
: "${MV15C_POST_JOB_ID:?}"
: "${MV15C_PYTHON:?}"
test -x "${MV15C_PYTHON}"
test -d "${MV15C_OUTPUT_ROOT}"

if [[ -f "${POINTER}" && "${MV15C_A1_ALLOW_RESUBMIT:-0}" != "1" ]]; then
  echo "Refusing duplicate MV15C-A1 submission; inspect ${POINTER}." >&2
  exit 3
fi
if [[ -e "${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV15C_A1_QY_RESULT.env" ]]; then
  echo "MV15C-A1 result already exists; refusing to overwrite it." >&2
  exit 4
fi

cd "${PROJECT_ROOT}"
export PATH="$(dirname "${MV15C_PYTHON}"):${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
"${MV15C_PYTHON}" -m vgdsmc.mohammadzadeh_mv15c_a1_qy_evaluation verify-amendment
"${MV15C_PYTHON}" -m vgdsmc.mohammadzadeh_mv15c_a1_qy_evaluation prepare-amendment \
  --output-root "${MV15C_OUTPUT_ROOT}"

mkdir -p "${PROJECT_ROOT}/logs"
EXPORTS="ALL,MV15C_A1_PROJECT_ROOT=${PROJECT_ROOT},MV15C_A1_PAYLOAD_ROOT=${PAYLOAD_ROOT},MV15C_A1_OUTPUT_ROOT=${MV15C_OUTPUT_ROOT},MV15C_A1_PYTHON=${MV15C_PYTHON},MV15C_A1_BATCH_SIZE=${MV15C_A1_BATCH_SIZE:-8},MV15C_A1_ORIGINAL_REFERENCE_JOB_ID=${MV15C_REFERENCE_JOB_ID},MV15C_A1_ORIGINAL_PREDICT_JOB_ID=${MV15C_PREDICT_JOB_ID},MV15C_A1_ORIGINAL_POST_JOB_ID=${MV15C_POST_JOB_ID}"
PREDICT_JOB="$(sbatch --parsable --export="${EXPORTS}" \
  --output="${PROJECT_ROOT}/logs/moh_mv15c_a1_predict_%j.slurm.out" \
  --error="${PROJECT_ROOT}/logs/moh_mv15c_a1_predict_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv15c_a1_predict.sbatch")"
PREDICT_JOB="${PREDICT_JOB%%;*}"
POST_JOB="$(sbatch --parsable --dependency="afterok:${PREDICT_JOB}" \
  --export="${EXPORTS},MV15C_A1_PREDICT_JOB_ID=${PREDICT_JOB}" \
  --output="${PROJECT_ROOT}/logs/moh_mv15c_a1_post_%j.slurm.out" \
  --error="${PROJECT_ROOT}/logs/moh_mv15c_a1_post_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv15c_a1_post.sbatch")"
POST_JOB="${POST_JOB%%;*}"

TEMPORARY="${POINTER}.tmp"
cat > "${TEMPORARY}" <<EOF
MV15C_A1_OUTPUT_ROOT=${MV15C_OUTPUT_ROOT}
MV15C_A1_PYTHON=${MV15C_PYTHON}
MV15C_A1_ORIGINAL_REFERENCE_JOB_ID=${MV15C_REFERENCE_JOB_ID}
MV15C_A1_ORIGINAL_PREDICT_JOB_ID=${MV15C_PREDICT_JOB_ID}
MV15C_A1_ORIGINAL_POST_JOB_ID=${MV15C_POST_JOB_ID}
MV15C_A1_PREDICT_JOB_ID=${PREDICT_JOB}
MV15C_A1_POST_JOB_ID=${POST_JOB}
MV15C_A1_JOB_IDS=${PREDICT_JOB},${POST_JOB}
EOF
mv "${TEMPORARY}" "${POINTER}"
echo "MV15C_A1_QY_SUBMITTED output=${MV15C_OUTPUT_ROOT} predict=${PREDICT_JOB} post=${POST_JOB} dsmc_rerun=NO"

