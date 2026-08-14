#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${MV15C_PROJECT_ROOT:-/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc}"
PAYLOAD_ROOT="${PROJECT_ROOT}/mv15c_fresh_b3_confirmation"
MV10_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV10_QY_JOB.env"
MV15B_JOB_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV15B_DATA_CONSISTENT_BUDGET_JOB.env"
MV15B_RESULT_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV15B_DATA_CONSISTENT_BUDGET_RESULT.env"
command -v sbatch >/dev/null
test -d "${PROJECT_ROOT}"
test -d "${PAYLOAD_ROOT}"
test -s "${MV10_POINTER}"
test -s "${MV15B_JOB_POINTER}"
test -s "${MV15B_RESULT_POINTER}"
source "${MV10_POINTER}"
source "${MV15B_JOB_POINTER}"
source "${MV15B_RESULT_POINTER}"
: "${MV10_MV9_OUTPUT_ROOT:?}"
: "${MV15B_OUTPUT_ROOT:?}"
: "${MV10_VENV_DIR:?}"
PYTHON_BIN="${MV15C_PYTHON:-${MV10_VENV_DIR}/bin/python}"
test -x "${PYTHON_BIN}"
"${PYTHON_BIN}" -c 'import matplotlib, numpy, torch' >/dev/null
test -s "${MV10_MV9_OUTPUT_ROOT}/dataset.npz"
test -s "${MV15B_OUTPUT_ROOT}/summary.json"
test -s "${MV15B_OUTPUT_ROOT}/selection_summary.json"
test -s "${MV15B_OUTPUT_ROOT}/locked_predictions.npz"
test -s "${MV15B_OUTPUT_ROOT}/artifact_manifest.json"

POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV15C_FRESH_B3_JOB.env"
if [[ -f "${POINTER}" && "${MV15C_ALLOW_NEW_RUN:-0}" != "1" ]]; then
  echo "Refusing duplicate MV15C submission; inspect ${POINTER}." >&2
  echo "Set MV15C_ALLOW_NEW_RUN=1 only for an intentional new confirmation." >&2
  exit 3
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
"${PYTHON_BIN}" -m vgdsmc.mohammadzadeh_mv15c_fresh_b3_confirmation verify-lock
mkdir -p "${PROJECT_ROOT}/logs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_ROOT="${PROJECT_ROOT}/results/mohammadzadeh_2012/mv15c_fresh_b3_confirmation/run_${STAMP}"
"${PYTHON_BIN}" -m vgdsmc.mohammadzadeh_mv15c_fresh_b3_confirmation prepare-lock \
  --mv9-output-root "${MV10_MV9_OUTPUT_ROOT}" \
  --mv15b-output-root "${MV15B_OUTPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}"

EXPORTS="ALL,MV15C_PROJECT_ROOT=${PROJECT_ROOT},MV15C_PAYLOAD_ROOT=${PAYLOAD_ROOT},MV15C_OUTPUT_ROOT=${OUTPUT_ROOT},MV15C_PYTHON=${PYTHON_BIN},MV15C_BATCH_SIZE=${MV15C_BATCH_SIZE:-8}"
REFERENCE_JOB="$(sbatch --parsable --export="${EXPORTS}" \
  --output="${PROJECT_ROOT}/logs/moh_mv15c_ref_%A_%a.slurm.out" \
  --error="${PROJECT_ROOT}/logs/moh_mv15c_ref_%A_%a.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv15c_reference.sbatch")"
REFERENCE_JOB="${REFERENCE_JOB%%;*}"
PREDICT_JOB="$(sbatch --parsable --dependency="afterok:${REFERENCE_JOB}" \
  --export="${EXPORTS},MV15C_REFERENCE_JOB_ID=${REFERENCE_JOB}" \
  --output="${PROJECT_ROOT}/logs/moh_mv15c_predict_%j.slurm.out" \
  --error="${PROJECT_ROOT}/logs/moh_mv15c_predict_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv15c_predict.sbatch")"
PREDICT_JOB="${PREDICT_JOB%%;*}"
POST_JOB="$(sbatch --parsable --dependency="afterok:${PREDICT_JOB}" \
  --export="${EXPORTS},MV15C_REFERENCE_JOB_ID=${REFERENCE_JOB},MV15C_PREDICT_JOB_ID=${PREDICT_JOB}" \
  --output="${PROJECT_ROOT}/logs/moh_mv15c_post_%j.slurm.out" \
  --error="${PROJECT_ROOT}/logs/moh_mv15c_post_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv15c_post.sbatch")"
POST_JOB="${POST_JOB%%;*}"

cat > "${POINTER}" <<EOF
MV15C_OUTPUT_ROOT=${OUTPUT_ROOT}
MV15C_MV9_OUTPUT_ROOT=${MV10_MV9_OUTPUT_ROOT}
MV15C_MV15B_OUTPUT_ROOT=${MV15B_OUTPUT_ROOT}
MV15C_PYTHON=${PYTHON_BIN}
MV15C_REFERENCE_JOB_ID=${REFERENCE_JOB}
MV15C_PREDICT_JOB_ID=${PREDICT_JOB}
MV15C_POST_JOB_ID=${POST_JOB}
MV15C_JOB_IDS=${REFERENCE_JOB},${PREDICT_JOB},${POST_JOB}
EOF
echo "MV15C_FRESH_B3_SUBMITTED output=${OUTPUT_ROOT} reference=${REFERENCE_JOB} predict=${PREDICT_JOB} post=${POST_JOB}"
