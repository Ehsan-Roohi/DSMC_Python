#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${MV14_PROJECT_ROOT:-/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc}"
PAYLOAD_ROOT="${PROJECT_ROOT}/mv14_kinetic_conservation_cavity"
MV10_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV10_QY_JOB.env"
MV12_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV12_SAGE_QY_JOB.env"
command -v sbatch >/dev/null
test -d "${PROJECT_ROOT}"
test -d "${PAYLOAD_ROOT}"
test -s "${MV10_POINTER}"
test -s "${MV12_POINTER}"
source "${MV10_POINTER}"
source "${MV12_POINTER}"
: "${MV10_MV9_OUTPUT_ROOT:?}"
: "${MV10_VENV_DIR:?}"
: "${MV12_OUTPUT_ROOT:?}"
test -s "${MV10_MV9_OUTPUT_ROOT}/dataset.npz"
test -s "${MV10_MV9_OUTPUT_ROOT}/source_moment_audit.csv"
test -s "${MV10_MV9_OUTPUT_ROOT}/summary.json"
test -s "${MV12_OUTPUT_ROOT}/summary.json"
test -s "${MV12_OUTPUT_ROOT}/artifact_manifest.json"
test -x "${MV10_VENV_DIR}/bin/python"

if [[ -f "${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV14_KINETIC_CAVITY_JOB.env" \
      && "${MV14_ALLOW_NEW_RUN:-0}" != "1" ]]; then
  echo "Refusing duplicate MV14 submission; inspect the existing pointer." >&2
  echo "Set MV14_ALLOW_NEW_RUN=1 only for an intentional rerun." >&2
  exit 3
fi

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
  "${MV10_VENV_DIR}/bin/python" -m vgdsmc.mohammadzadeh_mv14_kinetic_conservation_cavity \
  verify-data --mv9-output-root "${MV10_MV9_OUTPUT_ROOT}" \
  --mv12-output-root "${MV12_OUTPUT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_ROOT="${PROJECT_ROOT}/results/mohammadzadeh_2012/mv14_kinetic_conservation_cavity/run_${STAMP}"
EXPORTS="ALL,MV14_PROJECT_ROOT=${PROJECT_ROOT},MV14_PAYLOAD_ROOT=${PAYLOAD_ROOT},MV14_OUTPUT_ROOT=${OUTPUT_ROOT},MV14_MV9_OUTPUT_ROOT=${MV10_MV9_OUTPUT_ROOT},MV14_MV12_OUTPUT_ROOT=${MV12_OUTPUT_ROOT},MV14_VENV_DIR=${MV10_VENV_DIR},MV14_BATCH_SIZE=${MV14_BATCH_SIZE:-8}"

PREDICT_JOB=$(sbatch --parsable --export="${EXPORTS}" \
  --output="${PROJECT_ROOT}/logs/moh_mv14_kinetic_%j.slurm.out" \
  --error="${PROJECT_ROOT}/logs/moh_mv14_kinetic_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv14_kinetic_predict.sbatch")
PREDICT_JOB=${PREDICT_JOB%%;*}
POST_JOB=$(sbatch --parsable --dependency="afterok:${PREDICT_JOB}" \
  --export="${EXPORTS},MV14_PREDICT_JOB_ID=${PREDICT_JOB}" \
  --output="${PROJECT_ROOT}/logs/moh_mv14_post_%j.slurm.out" \
  --error="${PROJECT_ROOT}/logs/moh_mv14_post_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv14_kinetic_post.sbatch")
POST_JOB=${POST_JOB%%;*}

cat > "${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV14_KINETIC_CAVITY_JOB.env" <<EOF
MV14_OUTPUT_ROOT=${OUTPUT_ROOT}
MV14_MV9_OUTPUT_ROOT=${MV10_MV9_OUTPUT_ROOT}
MV14_MV12_OUTPUT_ROOT=${MV12_OUTPUT_ROOT}
MV14_VENV_DIR=${MV10_VENV_DIR}
MV14_PREDICT_JOB_ID=${PREDICT_JOB}
MV14_POST_JOB_ID=${POST_JOB}
MV14_JOB_IDS=${PREDICT_JOB},${POST_JOB}
EOF
echo "MV14_KINETIC_CAVITY_SUBMITTED output=${OUTPUT_ROOT} predict=${PREDICT_JOB} post=${POST_JOB}"
