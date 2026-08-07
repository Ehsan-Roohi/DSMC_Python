#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${MV2_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
M3_ROOT="${MV2_M3_ROOT:-${REPO_ROOT}/results/mohammadzadeh_2012/m3_qy_precision}"
OUTPUT_ROOT="${MV2_OUTPUT_ROOT:-${REPO_ROOT}/results/mohammadzadeh_2012/mv2_jcp_benchmark}"
PYTHON_BIN="${MV2_PYTHON:-python3}"

cd "${REPO_ROOT}"
for seed in 91901 91902 91903 91904 91905 91906 91907 91908; do
  test -f "${M3_ROOT}/seed_${seed}/block_fields.npz"
  test -f "${M3_ROOT}/seed_${seed}/fields.npz"
done

if [[ -e "${OUTPUT_ROOT}/summary.json" ]]; then
  echo "Refusing to overwrite completed MV2 output: ${OUTPUT_ROOT}" >&2
  exit 2
fi

mkdir -p logs "${OUTPUT_ROOT}"
VENV_DIR="${MV2_VENV_DIR:-}"
if [[ -z "${VENV_DIR}" ]]; then
  for candidate in "${REPO_ROOT}/.venv-mv2" "${REPO_ROOT}/.venv-mv1"; do
    if [[ -x "${candidate}/bin/python" ]] && \
       "${candidate}/bin/python" -c 'import numpy, torch, matplotlib' 2>/dev/null; then
      VENV_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${VENV_DIR}" ]]; then
  VENV_DIR="${REPO_ROOT}/.venv-mv2"
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
  source "${VENV_DIR}/bin/activate"
  python -m pip install -e '.[ml,plot]'
else
  source "${VENV_DIR}/bin/activate"
  python -m pip install -e . --no-deps
fi

EXPORTS="ALL,MV2_REPO_ROOT=${REPO_ROOT},MV2_M3_ROOT=${M3_ROOT},MV2_OUTPUT_ROOT=${OUTPUT_ROOT},MV2_VENV_DIR=${VENV_DIR}"
ARRAY_JOB_ID="$(sbatch --parsable --export="${EXPORTS}" scripts/unity_mohammadzadeh_vision_mv2_task.sbatch)"
POST_JOB_ID="$(sbatch --parsable --dependency="afterok:${ARRAY_JOB_ID}" --export="${EXPORTS}" scripts/unity_mohammadzadeh_vision_mv2_post.sbatch)"

ENV_FILE="${REPO_ROOT}/LAST_MOHAMMADZADEH_VISION_MV2_JOB.env"
printf 'ARRAY_JOB_ID=%q\nPOST_JOB_ID=%q\nMV2_REPO_ROOT=%q\nMV2_M3_ROOT=%q\nMV2_OUTPUT_ROOT=%q\nMV2_VENV_DIR=%q\n' \
  "${ARRAY_JOB_ID}" "${POST_JOB_ID}" "${REPO_ROOT}" "${M3_ROOT}" "${OUTPUT_ROOT}" "${VENV_DIR}" > "${ENV_FILE}"

echo "Submitted MV2 array: ${ARRAY_JOB_ID} (32 tasks, at most 4 concurrent)"
echo "Submitted MV2 postprocessor: ${POST_JOB_ID}"
echo "Saved: ${ENV_FILE}"
echo "Monitor: squeue -j ${ARRAY_JOB_ID},${POST_JOB_ID}"
