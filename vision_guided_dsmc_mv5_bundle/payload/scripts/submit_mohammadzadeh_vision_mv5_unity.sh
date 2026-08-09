#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${MV5_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
M3_ROOT="${MV5_M3_ROOT:-${REPO_ROOT}/results/mohammadzadeh_2012/m3_qy_precision}"
MV3_ROOT="${MV5_MV3_ROOT:-${REPO_ROOT}/results/mohammadzadeh_2012/mv3_cross_condition}"
MV4_ROOT="${MV5_MV4_ROOT:-${REPO_ROOT}/results/mohammadzadeh_2012/mv4_safe_reconstruction}"
OUTPUT_ROOT="${MV5_OUTPUT_ROOT:-${REPO_ROOT}/results/mohammadzadeh_2012/mv5_confirmatory_selector}"
PYTHON_BIN="${MV5_PYTHON:-python3}"

cd "${REPO_ROOT}"
python - "${MV3_ROOT}/verification.json" "${MV4_ROOT}/verification.json" \
  "${REPO_ROOT}/reference_data/mohammadzadeh_2012/mv5_confirmatory_protocol.json" <<'PY'
import json
import sys
from pathlib import Path

mv3 = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
mv4 = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
protocol = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
source = protocol["source_contract"]
if mv3.get("status") != source["mv3_required_verification_status"] or mv3.get("summary_sha256") != source["mv3_summary_sha256"]:
    raise SystemExit("MV5 requires the exact recursively verified MV3 source")
if mv4.get("status") != source["mv4_required_verification_status"] or mv4.get("summary_sha256") != source["mv4_summary_sha256"]:
    raise SystemExit("MV5 requires the exact recursively verified MV4 development outcome")
PY

for seed in 91901 91902 91903 91904; do
  test -f "${M3_ROOT}/seed_${seed}/block_fields.npz"
  test -f "${M3_ROOT}/seed_${seed}/fields.npz"
done

if [[ -e "${OUTPUT_ROOT}/summary.json" ]]; then
  echo "Refusing to overwrite completed MV5 output: ${OUTPUT_ROOT}" >&2
  exit 2
fi

mkdir -p logs "${OUTPUT_ROOT}"
VENV_DIR="${MV5_VENV_DIR:-}"
if [[ -z "${VENV_DIR}" && -f "${REPO_ROOT}/LAST_MOHAMMADZADEH_VISION_MV4_JOB.env" ]]; then
  source "${REPO_ROOT}/LAST_MOHAMMADZADEH_VISION_MV4_JOB.env"
  VENV_DIR="${MV4_VENV_DIR:-}"
fi
if [[ -z "${VENV_DIR}" && -f "${REPO_ROOT}/LAST_MOHAMMADZADEH_VISION_MV3_JOB.env" ]]; then
  source "${REPO_ROOT}/LAST_MOHAMMADZADEH_VISION_MV3_JOB.env"
  VENV_DIR="${MV3_VENV_DIR:-}"
fi
if [[ -z "${VENV_DIR}" ]]; then
  for candidate in "${REPO_ROOT}/.venv-mv4" "${REPO_ROOT}/.venv-mv3" "${REPO_ROOT}/.venv-mv2"; do
    if [[ -x "${candidate}/bin/python" ]] && \
       "${candidate}/bin/python" -c 'import numpy, torch, matplotlib' 2>/dev/null; then
      VENV_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${VENV_DIR}" ]]; then
  VENV_DIR="${REPO_ROOT}/.venv-mv5"
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
  source "${VENV_DIR}/bin/activate"
  python -m pip install -e '.[ml,plot]'
else
  source "${VENV_DIR}/bin/activate"
  python -m pip install -e . --no-deps
fi

python -m compileall -q \
  vgdsmc/mohammadzadeh_mv5_reference.py \
  vgdsmc/mohammadzadeh_vision_mv5.py
if python -c 'import pytest' 2>/dev/null; then
  python -m pytest -q tests/test_mohammadzadeh_vision_mv5.py
fi
python -m vgdsmc.mohammadzadeh_mv5_reference --verify-lock-only >/dev/null

EXPORTS="ALL,MV5_REPO_ROOT=${REPO_ROOT},MV5_M3_ROOT=${M3_ROOT},MV5_MV3_ROOT=${MV3_ROOT},MV5_MV4_ROOT=${MV4_ROOT},MV5_OUTPUT_ROOT=${OUTPUT_ROOT},MV5_VENV_DIR=${VENV_DIR},MV5_EPOCHS=${MV5_EPOCHS:-200},MV5_BATCH_SIZE=${MV5_BATCH_SIZE:-6}"
REFERENCE_JOB_ID="$(sbatch --parsable --export="${EXPORTS}" scripts/unity_mohammadzadeh_mv5_reference.sbatch)"
MODEL_JOB_ID="$(sbatch --parsable --dependency="afterok:${REFERENCE_JOB_ID}" --export="${EXPORTS}" scripts/unity_mohammadzadeh_vision_mv5_task.sbatch)"
POST_JOB_ID="$(sbatch --parsable --dependency="afterok:${MODEL_JOB_ID}" --export="${EXPORTS}" scripts/unity_mohammadzadeh_vision_mv5_post.sbatch)"

ENV_FILE="${REPO_ROOT}/LAST_MOHAMMADZADEH_VISION_MV5_JOB.env"
printf 'MV5_REFERENCE_JOB_ID=%q\nMV5_MODEL_JOB_ID=%q\nMV5_POST_JOB_ID=%q\nMV5_REPO_ROOT=%q\nMV5_M3_ROOT=%q\nMV5_MV3_ROOT=%q\nMV5_MV4_ROOT=%q\nMV5_OUTPUT_ROOT=%q\nMV5_VENV_DIR=%q\n' \
  "${REFERENCE_JOB_ID}" "${MODEL_JOB_ID}" "${POST_JOB_ID}" "${REPO_ROOT}" "${M3_ROOT}" "${MV3_ROOT}" "${MV4_ROOT}" "${OUTPUT_ROOT}" "${VENV_DIR}" > "${ENV_FILE}"

echo "Submitted MV5 confirmatory references: ${REFERENCE_JOB_ID} (16 tasks, at most 4 concurrent)"
echo "Submitted MV5 selector/model benchmark: ${MODEL_JOB_ID} (4 budgets)"
echo "Submitted MV5 postprocessor: ${POST_JOB_ID}"
echo "Saved: ${ENV_FILE}"
echo "Monitor: squeue -j ${REFERENCE_JOB_ID},${MODEL_JOB_ID},${POST_JOB_ID}"
