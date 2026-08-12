#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${MV7_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

test -f LAST_MOHAMMADZADEH_VISION_MV5_JOB.env
test -f LAST_MOHAMMADZADEH_MV5_REPAIR_MV6_JOB.env

source LAST_MOHAMMADZADEH_VISION_MV5_JOB.env
M3_ROOT="${MV7_M3_ROOT:-${MV5_M3_ROOT}}"
MV3_ROOT="${MV7_MV3_ROOT:-${MV5_MV3_ROOT}}"
SOURCE_REFERENCE_JOB_ID="${MV5_REFERENCE_JOB_ID:-}"

source LAST_MOHAMMADZADEH_MV5_REPAIR_MV6_JOB.env
REFERENCE_ROOT="${MV7_REFERENCE_ROOT:-${MV5R_ASSEMBLED_ROOT}}"
MV6_ROOT="${MV7_MV6_ROOT:-${MV6_OUTPUT_ROOT}}"
VENV_DIR="${MV7_VENV_DIR:-${MV6_VENV_DIR}}"
SOURCE_MV6_MODEL_JOB_ID="${MV6_MODEL_JOB_ID:-}"

test -x "${VENV_DIR}/bin/python"
test -f "${MV6_ROOT}/summary.json"
test -f "${MV6_ROOT}/verification.json"
test -d "${REFERENCE_ROOT}/references"

MV6_TASK_COUNT="$(find "${MV6_ROOT}/tasks" -mindepth 3 -maxdepth 3 \
  -name summary.json -type f | wc -l)"
if [[ "${MV6_TASK_COUNT}" -ne 12 ]]; then
  echo "MV7 requires exactly 12 completed MV6 budget-one tasks; found ${MV6_TASK_COUNT}" >&2
  exit 2
fi

if [[ -f "${REPO_ROOT}/LAST_MOHAMMADZADEH_MV7_JCP_JOB.env" \
      && "${MV7_ALLOW_NEW_RUN:-0}" != "1" ]]; then
  echo "Refusing duplicate MV7 submission; LAST_MOHAMMADZADEH_MV7_JCP_JOB.env already exists" >&2
  echo "Inspect the recorded run first, or set MV7_ALLOW_NEW_RUN=1 intentionally" >&2
  exit 3
fi

RUN_TAG="${MV7_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${MV7_OUTPUT_ROOT_OVERRIDE:-${REPO_ROOT}/results/mohammadzadeh_2012/mv7_jcp_budget_matrix/run_${RUN_TAG}}"
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite existing MV7 output: ${OUTPUT_ROOT}" >&2
  exit 4
fi
mkdir -p "${OUTPUT_ROOT}" logs

source "${VENV_DIR}/bin/activate"
python -m pip install -e . --no-deps
python -m compileall -q vgdsmc/mohammadzadeh_mv7_jcp_budget_matrix.py
python -m vgdsmc.mohammadzadeh_mv7_jcp_budget_matrix \
  --mode verify-lock >/dev/null
if python -c 'import pytest' 2>/dev/null; then
  python -m pytest -q tests/test_mohammadzadeh_mv7_jcp_budget_matrix.py
fi

COMMON_EXPORTS="ALL,MV7_REPO_ROOT=${REPO_ROOT},MV7_M3_ROOT=${M3_ROOT},MV7_MV3_ROOT=${MV3_ROOT},MV7_REFERENCE_ROOT=${REFERENCE_ROOT},MV7_MV6_ROOT=${MV6_ROOT},MV7_OUTPUT_ROOT=${OUTPUT_ROOT},MV7_VENV_DIR=${VENV_DIR},MV7_EPOCHS=${MV7_EPOCHS:-200},MV7_BATCH_SIZE=${MV7_BATCH_SIZE:-6}"

BASELINE_JOB_ID="$(sbatch --parsable --export="${COMMON_EXPORTS}" \
  scripts/unity_mohammadzadeh_mv7_baseline.sbatch)"
MODEL_JOB_ID="$(sbatch --parsable --dependency="afterok:${BASELINE_JOB_ID}" \
  --export="${COMMON_EXPORTS}" \
  scripts/unity_mohammadzadeh_mv7_model.sbatch)"
POST_EXPORTS="${COMMON_EXPORTS},MV7_BASELINE_JOB_ID=${BASELINE_JOB_ID},MV7_MODEL_JOB_ID=${MODEL_JOB_ID},MV7_SOURCE_MV6_MODEL_JOB_ID=${SOURCE_MV6_MODEL_JOB_ID},MV7_SOURCE_REFERENCE_JOB_ID=${SOURCE_REFERENCE_JOB_ID}"
POST_JOB_ID="$(sbatch --parsable --dependency="afterok:${MODEL_JOB_ID}" \
  --export="${POST_EXPORTS}" \
  scripts/unity_mohammadzadeh_mv7_post.sbatch)"

ENV_FILE="${REPO_ROOT}/LAST_MOHAMMADZADEH_MV7_JCP_JOB.env"
printf 'MV7_BASELINE_JOB_ID=%q\nMV7_MODEL_JOB_ID=%q\nMV7_POST_JOB_ID=%q\nMV7_REPO_ROOT=%q\nMV7_M3_ROOT=%q\nMV7_MV3_ROOT=%q\nMV7_REFERENCE_ROOT=%q\nMV7_MV6_ROOT=%q\nMV7_OUTPUT_ROOT=%q\nMV7_VENV_DIR=%q\nMV7_SOURCE_MV6_MODEL_JOB_ID=%q\nMV7_SOURCE_REFERENCE_JOB_ID=%q\n' \
  "${BASELINE_JOB_ID}" "${MODEL_JOB_ID}" "${POST_JOB_ID}" \
  "${REPO_ROOT}" "${M3_ROOT}" "${MV3_ROOT}" "${REFERENCE_ROOT}" \
  "${MV6_ROOT}" "${OUTPUT_ROOT}" "${VENV_DIR}" \
  "${SOURCE_MV6_MODEL_JOB_ID}" "${SOURCE_REFERENCE_JOB_ID}" > "${ENV_FILE}"

echo "Submitted MV7 classical baseline curves: ${BASELINE_JOB_ID} (4 budgets)"
echo "Submitted MV7 neural matrix: ${MODEL_JOB_ID} (36 new tasks, at most 4 concurrent)"
echo "Submitted MV7 locked JCP analysis: ${POST_JOB_ID}"
echo "Reused MV6 budget-one tasks from: ${MV6_ROOT}"
echo "Results: ${OUTPUT_ROOT}"
echo "Saved: ${ENV_FILE}"
echo "Monitor: squeue -j ${BASELINE_JOB_ID},${MODEL_JOB_ID},${POST_JOB_ID}"
