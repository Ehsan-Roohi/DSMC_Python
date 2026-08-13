#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${MV10_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

test -f LAST_MOHAMMADZADEH_MV9_HEAT_FLUX_JOB.env
set -a
source LAST_MOHAMMADZADEH_MV9_HEAT_FLUX_JOB.env
set +a

test -d "${MV9_OUTPUT_ROOT:?}"
test -f "${MV9_OUTPUT_ROOT}/dataset.npz"
test -f "${MV9_OUTPUT_ROOT}/summary.json"
test -f "${MV9_OUTPUT_ROOT}/verification.json"
test -x "${MV9_VENV_DIR:?}/bin/python"

if [[ -f LAST_MOHAMMADZADEH_MV10_QY_JOB.env \
      && "${MV10_ALLOW_NEW_RUN:-0}" != "1" ]]; then
  echo "Refusing duplicate MV10 submission; inspect LAST_MOHAMMADZADEH_MV10_QY_JOB.env" >&2
  echo "Set MV10_ALLOW_NEW_RUN=1 only for an intentional new method-development run" >&2
  exit 3
fi

RUN_TAG="${MV10_RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${MV10_OUTPUT_ROOT_OVERRIDE:-${REPO_ROOT}/results/mohammadzadeh_2012/mv10_qy_multiscale/run_${RUN_TAG}}"
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to overwrite MV10 output: ${OUTPUT_ROOT}" >&2
  exit 4
fi
mkdir -p logs

source "${MV9_VENV_DIR}/bin/activate"
python -m pip install -e . --no-deps
python -m compileall -q vgdsmc/mohammadzadeh_mv10_qy_multiscale.py
python -m vgdsmc.mohammadzadeh_mv10_qy_multiscale --mode verify-lock >/dev/null
if python -c 'import pytest' 2>/dev/null; then
  python -m pytest -q tests/test_mohammadzadeh_mv10_qy_multiscale.py
else
  echo "pytest is absent; continuing with prevalidated MV10 code."
fi

COMMON_EXPORTS="ALL,MV10_REPO_ROOT=${REPO_ROOT},MV10_MV9_OUTPUT_ROOT=${MV9_OUTPUT_ROOT},MV10_OUTPUT_ROOT=${OUTPUT_ROOT},MV10_VENV_DIR=${MV9_VENV_DIR},MV10_EPOCHS=${MV10_EPOCHS:-240},MV10_BATCH_SIZE=${MV10_BATCH_SIZE:-8}"

ASSEMBLE_JOB_ID="$(sbatch --parsable --export="${COMMON_EXPORTS}" \
  scripts/unity_mohammadzadeh_mv10_qy_assemble.sbatch)"
MODEL_JOB_ID="$(sbatch --parsable --dependency="afterok:${ASSEMBLE_JOB_ID}" \
  --export="${COMMON_EXPORTS}" scripts/unity_mohammadzadeh_mv10_qy_model.sbatch)"
POST_JOB_ID="$(sbatch --parsable --dependency="afterok:${MODEL_JOB_ID}" \
  --export="${COMMON_EXPORTS},MV10_ASSEMBLE_JOB_ID=${ASSEMBLE_JOB_ID},MV10_MODEL_JOB_ID=${MODEL_JOB_ID}" \
  scripts/unity_mohammadzadeh_mv10_qy_post.sbatch)"

ENV_FILE="${REPO_ROOT}/LAST_MOHAMMADZADEH_MV10_QY_JOB.env"
printf 'MV10_ASSEMBLE_JOB_ID=%q\nMV10_MODEL_JOB_ID=%q\nMV10_POST_JOB_ID=%q\nMV10_REPO_ROOT=%q\nMV10_MV9_OUTPUT_ROOT=%q\nMV10_OUTPUT_ROOT=%q\nMV10_VENV_DIR=%q\n' \
  "${ASSEMBLE_JOB_ID}" "${MODEL_JOB_ID}" "${POST_JOB_ID}" \
  "${REPO_ROOT}" "${MV9_OUTPUT_ROOT}" "${OUTPUT_ROOT}" "${MV9_VENV_DIR}" > "${ENV_FILE}"

echo "Submitted MV10 MV9-artifact verification/assembly: ${ASSEMBLE_JOB_ID}"
echo "Submitted MV10 qy local-coarse-global ensemble: ${MODEL_JOB_ID} (3 tasks)"
echo "Submitted MV10 legacy diagnostic/archive: ${POST_JOB_ID}"
echo "Results: ${OUTPUT_ROOT}"
echo "The compact ZIP will be written directly under: ${REPO_ROOT}"
echo "Saved: ${ENV_FILE}"
echo "Monitor: squeue -j ${ASSEMBLE_JOB_ID},${MODEL_JOB_ID},${POST_JOB_ID}"
