#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${MV12_PROJECT_ROOT:-/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc}"
PAYLOAD_ROOT="${PROJECT_ROOT}/mv12_sage_qy"
MV10_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV10_QY_JOB.env"
command -v sbatch >/dev/null
test -d "${PROJECT_ROOT}"
test -d "${PAYLOAD_ROOT}"
test -s "${MV10_POINTER}"
source "${MV10_POINTER}"
: "${MV10_OUTPUT_ROOT:?}"
: "${MV10_VENV_DIR:?}"
test -s "${MV10_OUTPUT_ROOT}/summary.json"
test -s "${MV10_OUTPUT_ROOT}/verification.json"
test -s "${MV10_OUTPUT_ROOT}/dataset.npz"
test -x "${MV10_VENV_DIR}/bin/python"
"${MV10_VENV_DIR}/bin/python" -c \
  'import torch; print("MV12_TORCH_PREFLIGHT_PASS", torch.__version__)'
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
  "${MV10_VENV_DIR}/bin/python" -m vgdsmc.mohammadzadeh_mv12_sage_qy \
  verify-data --mv10-output-root "${MV10_OUTPUT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_ROOT="${PROJECT_ROOT}/results/mohammadzadeh_2012/mv12_sage_qy/run_${STAMP}"
EXPORTS="ALL,MV12_PROJECT_ROOT=${PROJECT_ROOT},MV12_OUTPUT_ROOT=${OUTPUT_ROOT},MV12_MV10_OUTPUT_ROOT=${MV10_OUTPUT_ROOT},MV12_PAYLOAD_ROOT=${PAYLOAD_ROOT},MV12_VENV_DIR=${MV10_VENV_DIR}"

PREDICT_JOB=$(sbatch --parsable --export="${EXPORTS}" \
  --output="${PROJECT_ROOT}/logs/moh_mv12_sage_%j.slurm.out" \
  --error="${PROJECT_ROOT}/logs/moh_mv12_sage_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv12_sage_predict.sbatch")
PREDICT_JOB=${PREDICT_JOB%%;*}
JOB_IDS="${PREDICT_JOB}"
POST_EXPORTS="${EXPORTS},MV12_PREDICT_JOB_ID=${PREDICT_JOB}"
POST_JOB=$(sbatch --parsable --dependency="afterok:${PREDICT_JOB}" --export="${POST_EXPORTS}" \
  --output="${PROJECT_ROOT}/logs/moh_mv12_post_%j.slurm.out" \
  --error="${PROJECT_ROOT}/logs/moh_mv12_post_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv12_sage_post.sbatch")
POST_JOB=${POST_JOB%%;*}
JOB_IDS="${PREDICT_JOB},${POST_JOB}"

cat > "${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV12_SAGE_QY_JOB.env" <<EOF
MV12_OUTPUT_ROOT=${OUTPUT_ROOT}
MV12_MV10_OUTPUT_ROOT=${MV10_OUTPUT_ROOT}
MV12_VENV_DIR=${MV10_VENV_DIR}
MV12_PREDICT_JOB_ID=${PREDICT_JOB}
MV12_POST_JOB_ID=${POST_JOB}
MV12_JOB_IDS=${JOB_IDS}
EOF
echo "MV12_SAGE_SUBMITTED output=${OUTPUT_ROOT} predict=${PREDICT_JOB} post=${POST_JOB}"
