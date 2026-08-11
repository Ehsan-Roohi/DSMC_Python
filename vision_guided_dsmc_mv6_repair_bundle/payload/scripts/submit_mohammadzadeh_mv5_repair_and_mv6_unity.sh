#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${MV5R_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

source LAST_MOHAMMADZADEH_VISION_MV5_JOB.env
source LAST_MOHAMMADZADEH_ARCHITECTURE_SCREEN_JOB.env

ORIGINAL_ROOT="${MV5R_ORIGINAL_ROOT:-${MV5_OUTPUT_ROOT}}"
ASSEMBLED_ROOT="${MV5R_ASSEMBLED_ROOT:-${ORIGINAL_ROOT}_repaired}"
MV6_REPAIRED_OUTPUT_ROOT="${MV5R_MV6_OUTPUT_ROOT:-${REPO_ROOT}/results/mohammadzadeh_2012/mv6_architecture_screen_repaired_refs}"
VENV_DIR="${MV5R_VENV_DIR:-${MV6_VENV_DIR:-${MV5_VENV_DIR}}}"

test -x "${VENV_DIR}/bin/python"
test -f "${REPO_ROOT}/vgdsmc/mohammadzadeh_architecture_screen.py"
test -f "${REPO_ROOT}/scripts/unity_mohammadzadeh_architecture_screen_task.sbatch"
test -f "${REPO_ROOT}/scripts/unity_mohammadzadeh_architecture_screen_post.sbatch"

if find "${MV5_OUTPUT_ROOT}/tasks" "${MV6_OUTPUT_ROOT}/tasks" \
  -name summary.json -type f -print -quit 2>/dev/null | grep -q .; then
  echo "Refusing repair: a failed MV5/MV6 attempt produced a model summary" >&2
  exit 2
fi
if [[ -e "${MV6_REPAIRED_OUTPUT_ROOT}/summary.json" ]]; then
  echo "Refusing to overwrite completed repaired-reference MV6 output" >&2
  exit 3
fi

source "${VENV_DIR}/bin/activate"
python -m compileall -q \
  vgdsmc/mohammadzadeh_mv5_reference_stability_repair.py \
  vgdsmc/mohammadzadeh_architecture_screen.py
python -m vgdsmc.mohammadzadeh_mv5_reference_stability_repair \
  --mode verify-lock >/dev/null
if python -c 'import pytest' 2>/dev/null; then
  python -m pytest -q tests/test_mohammadzadeh_mv5_reference_stability_repair.py
fi

mkdir -p logs "${MV6_REPAIRED_OUTPUT_ROOT}"
REPAIR_EXPORTS="ALL,MV5R_REPO_ROOT=${REPO_ROOT},MV5R_ORIGINAL_ROOT=${ORIGINAL_ROOT},MV5R_ASSEMBLED_ROOT=${ASSEMBLED_ROOT},MV5R_VENV_DIR=${VENV_DIR}"
REPAIR_JOB_ID="$(sbatch --parsable --export="${REPAIR_EXPORTS}" scripts/unity_mohammadzadeh_mv5_reference_repair.sbatch)"
ASSEMBLE_JOB_ID="$(sbatch --parsable --dependency="afterok:${REPAIR_JOB_ID}" --export="${REPAIR_EXPORTS}" scripts/unity_mohammadzadeh_mv5_reference_repair_assemble.sbatch)"

MV6_EXPORTS="ALL,MV6_REPO_ROOT=${REPO_ROOT},MV6_M3_ROOT=${MV6_M3_ROOT},MV6_MV3_ROOT=${MV6_MV3_ROOT},MV6_MV5_REFERENCE_ROOT=${ASSEMBLED_ROOT},MV6_OUTPUT_ROOT=${MV6_REPAIRED_OUTPUT_ROOT},MV6_VENV_DIR=${VENV_DIR},MV6_EPOCHS=${MV6_EPOCHS:-200},MV6_BATCH_SIZE=${MV6_BATCH_SIZE:-6}"
MV6_MODEL_JOB_ID_NEW="$(sbatch --parsable --dependency="afterok:${ASSEMBLE_JOB_ID}" --export="${MV6_EXPORTS}" scripts/unity_mohammadzadeh_architecture_screen_task.sbatch)"
MV6_POST_JOB_ID_NEW="$(sbatch --parsable --dependency="afterok:${MV6_MODEL_JOB_ID_NEW}" --export="${MV6_EXPORTS}" scripts/unity_mohammadzadeh_architecture_screen_post.sbatch)"

ENV_FILE="${REPO_ROOT}/LAST_MOHAMMADZADEH_MV5_REPAIR_MV6_JOB.env"
printf 'MV5R_REPAIR_JOB_ID=%q\nMV5R_ASSEMBLE_JOB_ID=%q\nMV6_MODEL_JOB_ID=%q\nMV6_POST_JOB_ID=%q\nMV5R_ORIGINAL_ROOT=%q\nMV5R_ASSEMBLED_ROOT=%q\nMV6_OUTPUT_ROOT=%q\nMV6_VENV_DIR=%q\n' \
  "${REPAIR_JOB_ID}" "${ASSEMBLE_JOB_ID}" "${MV6_MODEL_JOB_ID_NEW}" "${MV6_POST_JOB_ID_NEW}" \
  "${ORIGINAL_ROOT}" "${ASSEMBLED_ROOT}" "${MV6_REPAIRED_OUTPUT_ROOT}" "${VENV_DIR}" > "${ENV_FILE}"

echo "Submitted MV5 late-window repairs: ${REPAIR_JOB_ID} (3 tasks)"
echo "Submitted repaired reference assembly: ${ASSEMBLE_JOB_ID}"
echo "Submitted MV6 architecture screen: ${MV6_MODEL_JOB_ID_NEW} (12 tasks, at most 4 concurrent)"
echo "Submitted MV6 postprocessor: ${MV6_POST_JOB_ID_NEW}"
echo "Saved: ${ENV_FILE}"
