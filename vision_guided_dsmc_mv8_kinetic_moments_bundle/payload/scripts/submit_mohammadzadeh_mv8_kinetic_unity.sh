#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${MV8_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

test -f LAST_MOHAMMADZADEH_MV7_JCP_JOB.env
set -a
source LAST_MOHAMMADZADEH_MV7_JCP_JOB.env
set +a

M3_ROOT="${MV8_M3_ROOT:-${MV7_M3_ROOT}}"
MV3_ROOT="${MV8_MV3_ROOT:-${MV7_MV3_ROOT}}"
REFERENCE_ROOT="${MV8_REFERENCE_ROOT:-${MV7_REFERENCE_ROOT}}"
VENV_DIR="${MV8_VENV_DIR:-${MV7_VENV_DIR}}"

test -x "${VENV_DIR}/bin/python"
test -d "${M3_ROOT}"
test -d "${MV3_ROOT}"
test -d "${REFERENCE_ROOT}/references"

if [[ -f LAST_MOHAMMADZADEH_MV8_KINETIC_JOB.env \
      && "${MV8_ALLOW_NEW_RUN:-0}" != "1" ]]; then
  echo "Refusing duplicate MV8 submission; inspect LAST_MOHAMMADZADEH_MV8_KINETIC_JOB.env" >&2
  echo "Set MV8_ALLOW_NEW_RUN=1 only when an intentional new pilot run is required" >&2
  exit 3
fi

RUN_TAG="${MV8_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${MV8_OUTPUT_ROOT_OVERRIDE:-${REPO_ROOT}/results/mohammadzadeh_2012/mv8_kinetic_moment_pilot/run_${RUN_TAG}}"
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite MV8 output: ${OUTPUT_ROOT}" >&2
  exit 4
fi
mkdir -p "${OUTPUT_ROOT}" logs

source "${VENV_DIR}/bin/activate"
python -m pip install -e . --no-deps
python -m compileall -q vgdsmc/mohammadzadeh_mv8_kinetic_moments.py
python -m vgdsmc.mohammadzadeh_mv8_kinetic_moments --mode verify-lock >/dev/null
if python -c 'import pytest' 2>/dev/null; then
  python -m pytest -q tests/test_mohammadzadeh_mv8_kinetic_moments.py
else
  echo "pytest is absent; continuing with prevalidated MV8 code."
fi

COMMON_EXPORTS="ALL,MV8_REPO_ROOT=${REPO_ROOT},MV8_M3_ROOT=${M3_ROOT},MV8_MV3_ROOT=${MV3_ROOT},MV8_REFERENCE_ROOT=${REFERENCE_ROOT},MV8_OUTPUT_ROOT=${OUTPUT_ROOT},MV8_VENV_DIR=${VENV_DIR},MV8_EPOCHS=${MV8_EPOCHS:-200},MV8_BATCH_SIZE=${MV8_BATCH_SIZE:-6}"

ASSEMBLE_JOB_ID="$(sbatch --parsable --export="${COMMON_EXPORTS}" \
  scripts/unity_mohammadzadeh_mv8_assemble.sbatch)"
MODEL_JOB_ID="$(sbatch --parsable --dependency="afterok:${ASSEMBLE_JOB_ID}" \
  --export="${COMMON_EXPORTS}" scripts/unity_mohammadzadeh_mv8_model.sbatch)"
POST_JOB_ID="$(sbatch --parsable --dependency="afterok:${MODEL_JOB_ID}" \
  --export="${COMMON_EXPORTS},MV8_ASSEMBLE_JOB_ID=${ASSEMBLE_JOB_ID},MV8_MODEL_JOB_ID=${MODEL_JOB_ID}" \
  scripts/unity_mohammadzadeh_mv8_post.sbatch)"

ENV_FILE="${REPO_ROOT}/LAST_MOHAMMADZADEH_MV8_KINETIC_JOB.env"
printf 'MV8_ASSEMBLE_JOB_ID=%q\nMV8_MODEL_JOB_ID=%q\nMV8_POST_JOB_ID=%q\nMV8_REPO_ROOT=%q\nMV8_M3_ROOT=%q\nMV8_MV3_ROOT=%q\nMV8_REFERENCE_ROOT=%q\nMV8_OUTPUT_ROOT=%q\nMV8_VENV_DIR=%q\n' \
  "${ASSEMBLE_JOB_ID}" "${MODEL_JOB_ID}" "${POST_JOB_ID}" \
  "${REPO_ROOT}" "${M3_ROOT}" "${MV3_ROOT}" "${REFERENCE_ROOT}" \
  "${OUTPUT_ROOT}" "${VENV_DIR}" > "${ENV_FILE}"

echo "Submitted MV8 additive-moment audit/assembly: ${ASSEMBLE_JOB_ID}"
echo "Submitted MV8 B=1 neural pilot: ${MODEL_JOB_ID} (6 tasks, at most 3 concurrent)"
echo "Submitted MV8 analysis/physical contours/archive: ${POST_JOB_ID}"
echo "Results: ${OUTPUT_ROOT}"
echo "Saved: ${ENV_FILE}"
echo "Monitor without tail -f: squeue -j ${ASSEMBLE_JOB_ID},${MODEL_JOB_ID},${POST_JOB_ID}"
