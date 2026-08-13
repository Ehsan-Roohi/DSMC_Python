#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${MV9_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

test -f LAST_MOHAMMADZADEH_MV7_JCP_JOB.env
set -a
source LAST_MOHAMMADZADEH_MV7_JCP_JOB.env
set +a

M3_ROOT="${MV9_M3_ROOT:-${MV7_M3_ROOT}}"
MV3_ROOT="${MV9_MV3_ROOT:-${MV7_MV3_ROOT}}"
REFERENCE_ROOT="${MV9_REFERENCE_ROOT:-${MV7_REFERENCE_ROOT}}"
VENV_DIR="${MV9_VENV_DIR:-${MV7_VENV_DIR}}"

test -x "${VENV_DIR}/bin/python"
test -d "${M3_ROOT}"
test -d "${MV3_ROOT}"
test -d "${REFERENCE_ROOT}/references"

if [[ -f LAST_MOHAMMADZADEH_MV9_HEAT_FLUX_JOB.env \
      && "${MV9_ALLOW_NEW_RUN:-0}" != "1" ]]; then
  echo "Refusing duplicate MV9 submission; inspect LAST_MOHAMMADZADEH_MV9_HEAT_FLUX_JOB.env" >&2
  echo "Set MV9_ALLOW_NEW_RUN=1 only when an intentional new pilot run is required" >&2
  exit 3
fi

RUN_TAG="${MV9_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${MV9_OUTPUT_ROOT_OVERRIDE:-${REPO_ROOT}/results/mohammadzadeh_2012/mv9_heat_flux_noise2noise/run_${RUN_TAG}}"
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite MV9 output: ${OUTPUT_ROOT}" >&2
  exit 4
fi
mkdir -p "${OUTPUT_ROOT}" logs

source "${VENV_DIR}/bin/activate"
python -m pip install -e . --no-deps
python -m compileall -q vgdsmc/mohammadzadeh_mv9_heat_flux.py
python -m vgdsmc.mohammadzadeh_mv9_heat_flux --mode verify-lock >/dev/null
if python -c 'import pytest' 2>/dev/null; then
  python -m pytest -q tests/test_mohammadzadeh_mv9_heat_flux.py
else
  echo "pytest is absent; continuing with prevalidated MV9 code."
fi

COMMON_EXPORTS="ALL,MV9_REPO_ROOT=${REPO_ROOT},MV9_M3_ROOT=${M3_ROOT},MV9_MV3_ROOT=${MV3_ROOT},MV9_REFERENCE_ROOT=${REFERENCE_ROOT},MV9_OUTPUT_ROOT=${OUTPUT_ROOT},MV9_VENV_DIR=${VENV_DIR},MV9_EPOCHS=${MV9_EPOCHS:-180},MV9_BATCH_SIZE=${MV9_BATCH_SIZE:-8}"

ASSEMBLE_JOB_ID="$(sbatch --parsable --export="${COMMON_EXPORTS}" \
  scripts/unity_mohammadzadeh_mv9_assemble.sbatch)"
MODEL_JOB_ID="$(sbatch --parsable --dependency="afterok:${ASSEMBLE_JOB_ID}" \
  --export="${COMMON_EXPORTS}" scripts/unity_mohammadzadeh_mv9_model.sbatch)"
POST_JOB_ID="$(sbatch --parsable --dependency="afterok:${MODEL_JOB_ID}" \
  --export="${COMMON_EXPORTS},MV9_ASSEMBLE_JOB_ID=${ASSEMBLE_JOB_ID},MV9_MODEL_JOB_ID=${MODEL_JOB_ID}" \
  scripts/unity_mohammadzadeh_mv9_post.sbatch)"

ENV_FILE="${REPO_ROOT}/LAST_MOHAMMADZADEH_MV9_HEAT_FLUX_JOB.env"
printf 'MV9_ASSEMBLE_JOB_ID=%q\nMV9_MODEL_JOB_ID=%q\nMV9_POST_JOB_ID=%q\nMV9_REPO_ROOT=%q\nMV9_M3_ROOT=%q\nMV9_MV3_ROOT=%q\nMV9_REFERENCE_ROOT=%q\nMV9_OUTPUT_ROOT=%q\nMV9_VENV_DIR=%q\n' \
  "${ASSEMBLE_JOB_ID}" "${MODEL_JOB_ID}" "${POST_JOB_ID}" \
  "${REPO_ROOT}" "${M3_ROOT}" "${MV3_ROOT}" "${REFERENCE_ROOT}" \
  "${OUTPUT_ROOT}" "${VENV_DIR}" > "${ENV_FILE}"

echo "Submitted MV9 additive-moment audit/assembly: ${ASSEMBLE_JOB_ID}"
echo "Submitted MV9 cross-seed Noise2Noise pilot: ${MODEL_JOB_ID} (6 tasks, at most 3 concurrent)"
echo "Submitted MV9 analysis/physical contours/archive: ${POST_JOB_ID}"
echo "Results: ${OUTPUT_ROOT}"
echo "Saved: ${ENV_FILE}"
echo "Monitor without tail -f: squeue -j ${ASSEMBLE_JOB_ID},${MODEL_JOB_ID},${POST_JOB_ID}"
