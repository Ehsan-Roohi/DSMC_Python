#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${M3_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUTPUT_ROOT="${M3_OUTPUT_ROOT:-${REPO_ROOT}/results/mohammadzadeh_2012/m3_qy_precision}"
PYTHON_BIN="${M3_PYTHON:-python3}"

cd "${REPO_ROOT}"
mkdir -p logs "${OUTPUT_ROOT}"
if [[ ! -x "${REPO_ROOT}/.venv-m3/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${REPO_ROOT}/.venv-m3"
fi
source "${REPO_ROOT}/.venv-m3/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[plot]'
python -m vgdsmc.mohammadzadeh_qy_precision --verify-lock-only \
  > "${OUTPUT_ROOT}/preflight.json"

EXPORTS="ALL,M3_REPO_ROOT=${REPO_ROOT},M3_OUTPUT_ROOT=${OUTPUT_ROOT}"
ARRAY_JOB_ID="$(sbatch --parsable --export="${EXPORTS}" scripts/unity_m3_qy_precision.sbatch)"
POST_JOB_ID="$(sbatch --parsable --dependency="afterok:${ARRAY_JOB_ID}" --export="${EXPORTS}" scripts/unity_m3_qy_postprocess.sbatch)"
ENV_FILE="${REPO_ROOT}/LAST_M3_QY_JOB.env"
printf 'ARRAY_JOB_ID=%q\nPOST_JOB_ID=%q\nM3_REPO_ROOT=%q\nM3_OUTPUT_ROOT=%q\n' \
  "${ARRAY_JOB_ID}" "${POST_JOB_ID}" "${REPO_ROOT}" "${OUTPUT_ROOT}" \
  > "${ENV_FILE}"
echo "Submitted M3 QY100 array job: ${ARRAY_JOB_ID}"
echo "Submitted dependent postprocess job: ${POST_JOB_ID}"
echo "Saved: ${ENV_FILE}"
echo "Monitor: squeue -j ${ARRAY_JOB_ID},${POST_JOB_ID}"
