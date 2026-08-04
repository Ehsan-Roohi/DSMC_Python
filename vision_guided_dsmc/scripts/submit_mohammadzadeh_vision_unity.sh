#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${MV1_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
M3_ROOT="${MV1_M3_ROOT:-${REPO_ROOT}/results/mohammadzadeh_2012/m3_qy_precision}"
OUTPUT_ROOT="${MV1_OUTPUT_ROOT:-${REPO_ROOT}/results/mohammadzadeh_2012/mv1_vision}"
PYTHON_BIN="${MV1_PYTHON:-python3}"

cd "${REPO_ROOT}"
for seed in 91901 91902 91903 91904 91905 91906 91907 91908; do
  test -f "${M3_ROOT}/seed_${seed}/block_fields.npz"
  test -f "${M3_ROOT}/seed_${seed}/fields.npz"
done
mkdir -p logs "${OUTPUT_ROOT}"
if [[ ! -x "${REPO_ROOT}/.venv-mv1/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${REPO_ROOT}/.venv-mv1"
fi
source "${REPO_ROOT}/.venv-mv1/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[ml,plot]'

EXPORTS="ALL,MV1_REPO_ROOT=${REPO_ROOT},MV1_M3_ROOT=${M3_ROOT},MV1_OUTPUT_ROOT=${OUTPUT_ROOT}"
JOB_ID="$(sbatch --parsable --export="${EXPORTS}" scripts/unity_mohammadzadeh_vision.sbatch)"
ENV_FILE="${REPO_ROOT}/LAST_MOHAMMADZADEH_VISION_JOB.env"
printf 'JOB_ID=%q\nMV1_REPO_ROOT=%q\nMV1_M3_ROOT=%q\nMV1_OUTPUT_ROOT=%q\n' \
  "${JOB_ID}" "${REPO_ROOT}" "${M3_ROOT}" "${OUTPUT_ROOT}" > "${ENV_FILE}"
echo "Submitted Mohammadzadeh MV1 job: ${JOB_ID}"
echo "Saved: ${ENV_FILE}"
echo "Monitor: squeue -j ${JOB_ID}"
