#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${MV4_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
M3_ROOT="${MV4_M3_ROOT:-${REPO_ROOT}/results/mohammadzadeh_2012/m3_qy_precision}"
REFERENCE_ROOT="${MV4_REFERENCE_ROOT:-${REPO_ROOT}/results/mohammadzadeh_2012/mv3_cross_condition}"
OUTPUT_ROOT="${MV4_OUTPUT_ROOT:-${REPO_ROOT}/results/mohammadzadeh_2012/mv4_safe_reconstruction}"
PYTHON_BIN="${MV4_PYTHON:-python3}"

cd "${REPO_ROOT}"
test -f "${REFERENCE_ROOT}/verification.json"
python - "${REFERENCE_ROOT}/verification.json" "${REPO_ROOT}/reference_data/mohammadzadeh_2012/mv4_stability_repair_protocol.json" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
protocol = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
source = protocol["source_contract"]
if value.get("status") != source["mv3_required_verification_status"]:
    raise SystemExit("MV4 requires the recursively verified MV3 artifact tree")
if value.get("summary_sha256") != source["mv3_summary_sha256"]:
    raise SystemExit("MV3 summary hash differs from the pre-outcome MV4 lock")
PY

for seed in 91901 91902 91903 91904 91905 91906 91907 91908; do
  test -f "${M3_ROOT}/seed_${seed}/block_fields.npz"
  test -f "${M3_ROOT}/seed_${seed}/fields.npz"
done

if [[ -e "${OUTPUT_ROOT}/summary.json" ]]; then
  echo "Refusing to overwrite completed MV4 output: ${OUTPUT_ROOT}" >&2
  exit 2
fi

mkdir -p logs "${OUTPUT_ROOT}"
VENV_DIR="${MV4_VENV_DIR:-}"
if [[ -z "${VENV_DIR}" && -f "${REPO_ROOT}/LAST_MOHAMMADZADEH_VISION_MV3_JOB.env" ]]; then
  source "${REPO_ROOT}/LAST_MOHAMMADZADEH_VISION_MV3_JOB.env"
  VENV_DIR="${MV3_VENV_DIR:-}"
fi
if [[ -z "${VENV_DIR}" ]]; then
  for candidate in "${REPO_ROOT}/.venv-mv3" "${REPO_ROOT}/.venv-mv2" "${REPO_ROOT}/.venv-mv1"; do
    if [[ -x "${candidate}/bin/python" ]] && \
       "${candidate}/bin/python" -c 'import numpy, torch, matplotlib' 2>/dev/null; then
      VENV_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${VENV_DIR}" ]]; then
  VENV_DIR="${REPO_ROOT}/.venv-mv4"
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
  source "${VENV_DIR}/bin/activate"
  python -m pip install -e '.[ml,plot]'
else
  source "${VENV_DIR}/bin/activate"
  python -m pip install -e . --no-deps
fi

python -m compileall -q vgdsmc/mohammadzadeh_vision_mv4.py
if python -c 'import pytest' 2>/dev/null; then
  python -m pytest -q tests/test_mohammadzadeh_vision_mv4.py
fi
python -m vgdsmc.mohammadzadeh_mv3_reference --verify-lock-only >/dev/null

EXPORTS="ALL,MV4_REPO_ROOT=${REPO_ROOT},MV4_M3_ROOT=${M3_ROOT},MV4_REFERENCE_ROOT=${REFERENCE_ROOT},MV4_OUTPUT_ROOT=${OUTPUT_ROOT},MV4_VENV_DIR=${VENV_DIR},MV4_EPOCHS=${MV4_EPOCHS:-160},MV4_BATCH_SIZE=${MV4_BATCH_SIZE:-6}"
MODEL_JOB_ID="$(sbatch --parsable --export="${EXPORTS}" scripts/unity_mohammadzadeh_vision_mv4_task.sbatch)"
POST_JOB_ID="$(sbatch --parsable --dependency="afterok:${MODEL_JOB_ID}" --export="${EXPORTS}" scripts/unity_mohammadzadeh_vision_mv4_post.sbatch)"

ENV_FILE="${REPO_ROOT}/LAST_MOHAMMADZADEH_VISION_MV4_JOB.env"
printf 'MV4_MODEL_JOB_ID=%q\nMV4_POST_JOB_ID=%q\nMV4_REPO_ROOT=%q\nMV4_M3_ROOT=%q\nMV4_REFERENCE_ROOT=%q\nMV4_OUTPUT_ROOT=%q\nMV4_VENV_DIR=%q\n' \
  "${MODEL_JOB_ID}" "${POST_JOB_ID}" "${REPO_ROOT}" "${M3_ROOT}" "${REFERENCE_ROOT}" "${OUTPUT_ROOT}" "${VENV_DIR}" > "${ENV_FILE}"

echo "Submitted MV4 bounded/safe model benchmark: ${MODEL_JOB_ID} (16 tasks, at most 4 concurrent)"
echo "Submitted MV4 postprocessor: ${POST_JOB_ID}"
echo "Saved: ${ENV_FILE}"
echo "Monitor: squeue -j ${MODEL_JOB_ID},${POST_JOB_ID}"
