#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${MV17B_PROJECT_ROOT:-/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc}"
PAYLOAD_ROOT="${PROJECT_ROOT}/mv17b_a1_mechanical_recovery"
ORIGINAL_JOB_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV17B_FRESH_CYLINDER_JOB.env"
A1_JOB_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV17B_A1_MECHANICAL_RECOVERY_JOB.env"
RESULT_POINTER="${PROJECT_ROOT}/LAST_MOHAMMADZADEH_MV17B_FRESH_CYLINDER_RESULT.env"
command -v sbatch >/dev/null
test -s "${ORIGINAL_JOB_POINTER}"
test ! -e "${A1_JOB_POINTER}"
test ! -e "${RESULT_POINTER}"
source "${ORIGINAL_JOB_POINTER}"
: "${MV17B_OUTPUT_ROOT:?}"
: "${MV17B_JOB_IDS:?}"
PYTHON_BIN="${MV17B_A1_PYTHON:-${MV17B_PYTHON:-}}"
test -x "${PYTHON_BIN}"
test ! -e "${MV17B_OUTPUT_ROOT}/analysis"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
find \
  "${PROJECT_ROOT}/vgdsmc/mohammadzadeh_mv17b_a1_mechanical_recovery.py" \
  "${PAYLOAD_ROOT}/scripts" \
  "${PAYLOAD_ROOT}/tests" \
  "${PROJECT_ROOT}/reference_data/mohammadzadeh_2012/mv17b_a1_mechanical_recovery_amendment.json" \
  -type f -print0 | sort -z | xargs -0 sha256sum \
  > "${MV17B_OUTPUT_ROOT}/MV17B_A1_CODE_SHA256SUMS.txt"
"${PYTHON_BIN}" -m vgdsmc.mohammadzadeh_mv17b_a1_mechanical_recovery lock \
  --output-root "${MV17B_OUTPUT_ROOT}" \
  --amendment "${PROJECT_ROOT}/reference_data/mohammadzadeh_2012/mv17b_a1_mechanical_recovery_amendment.json"

EXPORTS="ALL,MV17B_PROJECT_ROOT=${PROJECT_ROOT},MV17B_A1_PAYLOAD_ROOT=${PAYLOAD_ROOT},MV17B_OUTPUT_ROOT=${MV17B_OUTPUT_ROOT},MV17B_PYTHON=${PYTHON_BIN}"
RECOVERY_JOB=$(sbatch --parsable --array='1-4,9-11%4' --export="${EXPORTS}" \
  --output="${MV17B_OUTPUT_ROOT}/campaign/logs/recovery_%A_%a.slurm.out" \
  --error="${MV17B_OUTPUT_ROOT}/campaign/logs/recovery_%A_%a.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv17b_a1_recover_array.sbatch")
RECOVERY_JOB=${RECOVERY_JOB%%;*}
ANALYZE_JOB=$(sbatch --parsable --dependency="afterok:${RECOVERY_JOB}" --export="${EXPORTS}" \
  --output="${MV17B_OUTPUT_ROOT}/campaign/logs/analyze_a1_%j.slurm.out" \
  --error="${MV17B_OUTPUT_ROOT}/campaign/logs/analyze_a1_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv17b_a1_analyze.sbatch")
ANALYZE_JOB=${ANALYZE_JOB%%;*}

SUBMISSION="${MV17B_OUTPUT_ROOT}/MV17B_A1_SUBMISSION.env"
POST_EXPORTS="${EXPORTS},MV17B_A1_SUBMISSION_ENV=${SUBMISSION}"
POST_JOB=$(sbatch --parsable --dependency="afterok:${ANALYZE_JOB}" --export="${POST_EXPORTS}" \
  --output="${MV17B_OUTPUT_ROOT}/campaign/logs/post_a1_%j.slurm.out" \
  --error="${MV17B_OUTPUT_ROOT}/campaign/logs/post_a1_%j.slurm.err" \
  "${PAYLOAD_ROOT}/scripts/unity_mohammadzadeh_mv17b_a1_post.sbatch")
POST_JOB=${POST_JOB%%;*}
NEW_JOB_IDS="${RECOVERY_JOB},${ANALYZE_JOB},${POST_JOB}"
ALL_JOB_IDS="${MV17B_JOB_IDS},${NEW_JOB_IDS}"
cat > "${SUBMISSION}" <<EOF
MV17B_OUTPUT_ROOT=${MV17B_OUTPUT_ROOT}
MV17B_PYTHON=${PYTHON_BIN}
MV17B_A1_RECOVERY_JOB_ID=${RECOVERY_JOB}
MV17B_A1_ANALYZE_JOB_ID=${ANALYZE_JOB}
MV17B_A1_POST_JOB_ID=${POST_JOB}
MV17B_A1_JOB_IDS=${NEW_JOB_IDS}
MV17B_A1_ALL_JOB_IDS=${ALL_JOB_IDS}
MV17B_A1_RECOVERY_ARRAY=1-4,9-11
MV17B_A1_RECOVERY_TRAJECTORIES=7
MV17B_A1_COMPLETE_TRAJECTORIES_REUSED=5
MV17B_A1_SEED_REPLACEMENT=false
MV17B_A1_SCIENTIFIC_CONTRACT_CHANGED=false
EOF
cp "${SUBMISSION}" "${A1_JOB_POINTER}.tmp"
mv "${A1_JOB_POINTER}.tmp" "${A1_JOB_POINTER}"
echo "MV17B_A1_RECOVERY_SUBMITTED output=${MV17B_OUTPUT_ROOT} recovery=${RECOVERY_JOB} analyze=${ANALYZE_JOB} post=${POST_JOB} reruns=7 reused=5"
