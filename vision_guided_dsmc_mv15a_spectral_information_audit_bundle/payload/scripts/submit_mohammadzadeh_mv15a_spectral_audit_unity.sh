#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${MV15A_PROJECT_ROOT:-/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc}"
PAYLOAD_ROOT="${PROJECT_ROOT}/mv15a_spectral_information_audit"
MV10_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV10_QY_JOB.env"
MV14_JOB_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV14_KINETIC_CAVITY_JOB.env"
MV14_RESULT_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV14_KINETIC_CAVITY_RESULT.env"
command -v sbatch >/dev/null
test -d "${PROJECT_ROOT}"
test -d "${PAYLOAD_ROOT}"
test -s "${MV10_POINTER}"
test -s "${MV14_JOB_POINTER}"
test -s "${MV14_RESULT_POINTER}"
source "${MV10_POINTER}"
source "${MV14_JOB_POINTER}"
source "${MV14_RESULT_POINTER}"
: "${MV10_MV9_OUTPUT_ROOT:?}"
: "${MV10_VENV_DIR:?}"
: "${MV14_OUTPUT_ROOT:?}"
PYTHON_BIN="${MV15A_PYTHON:-${MV10_VENV_DIR}/bin/python}"
test -x "${PYTHON_BIN}"
"${PYTHON_BIN}" -c 'import numpy, torch' >/dev/null
test -s "${MV10_MV9_OUTPUT_ROOT}/dataset.npz"
test -s "${MV14_OUTPUT_ROOT}/summary.json"
test -s "${MV14_OUTPUT_ROOT}/locked_predictions.npz"
test -s "${MV14_OUTPUT_ROOT}/artifact_manifest.json"

if [[ -f "${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV15A_SPECTRAL_AUDIT_JOB.env" \
      && "${MV15A_ALLOW_NEW_RUN:-0}" != "1" ]]; then
  echo "Refusing duplicate MV15A submission; inspect the existing pointer." >&2
  echo "Set MV15A_ALLOW_NEW_RUN=1 only for an intentional rerun." >&2
  exit 3
fi

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" -m vgdsmc.mohammadzadeh_mv15a_spectral_information_audit \
  verify-data --mv9-output-root "${MV10_MV9_OUTPUT_ROOT}" \
  --mv14-output-root "${MV14_OUTPUT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_ROOT="${PROJECT_ROOT}/results/mohammadzadeh_2012/mv15a_spectral_information_audit/run_${STAMP}"
EXPORTS="ALL,MV15A_PROJECT_ROOT=${PROJECT_ROOT},MV15A_PAYLOAD_ROOT=${PAYLOAD_ROOT},MV15A_OUTPUT_ROOT=${OUTPUT_ROOT},MV15A_MV9_OUTPUT_ROOT=${MV10_MV9_OUTPUT_ROOT},MV15A_MV14_OUTPUT_ROOT=${MV14_OUTPUT_ROOT},MV15A_PYTHON=${PYTHON_BIN},MV15A_BATCH_SIZE=${MV15A_BATCH_SIZE:-8}"

PREDICT_JOB=$(sbatch --parsable --export="${EXPORTS}" \
  --output="${PROJECT_ROOT}/logs/moh_mv15a_spectral_%j.slurm.out" \
  --error="${PROJECT_ROOT}/logs/moh_mv15a_spectral_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv15a_spectral_predict.sbatch")
PREDICT_JOB=${PREDICT_JOB%%;*}
POST_JOB=$(sbatch --parsable --dependency="afterok:${PREDICT_JOB}" \
  --export="${EXPORTS},MV15A_PREDICT_JOB_ID=${PREDICT_JOB}" \
  --output="${PROJECT_ROOT}/logs/moh_mv15a_post_%j.slurm.out" \
  --error="${PROJECT_ROOT}/logs/moh_mv15a_post_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv15a_spectral_post.sbatch")
POST_JOB=${POST_JOB%%;*}

cat > "${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV15A_SPECTRAL_AUDIT_JOB.env" <<EOF
MV15A_OUTPUT_ROOT=${OUTPUT_ROOT}
MV15A_MV9_OUTPUT_ROOT=${MV10_MV9_OUTPUT_ROOT}
MV15A_MV14_OUTPUT_ROOT=${MV14_OUTPUT_ROOT}
MV15A_PYTHON=${PYTHON_BIN}
MV15A_PREDICT_JOB_ID=${PREDICT_JOB}
MV15A_POST_JOB_ID=${POST_JOB}
MV15A_JOB_IDS=${PREDICT_JOB},${POST_JOB}
EOF
echo "MV15A_SPECTRAL_AUDIT_SUBMITTED output=${OUTPUT_ROOT} predict=${PREDICT_JOB} post=${POST_JOB}"
