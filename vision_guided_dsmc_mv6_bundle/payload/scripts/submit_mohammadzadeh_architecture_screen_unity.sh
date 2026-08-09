#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${MV6_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
M3_ROOT="${MV6_M3_ROOT:-${REPO_ROOT}/results/mohammadzadeh_2012/m3_qy_precision}"
MV3_ROOT="${MV6_MV3_ROOT:-${REPO_ROOT}/results/mohammadzadeh_2012/mv3_cross_condition}"
MV5_REFERENCE_ROOT="${MV6_MV5_REFERENCE_ROOT:-${REPO_ROOT}/results/mohammadzadeh_2012/mv5_confirmatory_selector}"
OUTPUT_ROOT="${MV6_OUTPUT_ROOT:-${REPO_ROOT}/results/mohammadzadeh_2012/mv6_architecture_screen}"
PYTHON_BIN="${MV6_PYTHON:-python3}"

cd "${REPO_ROOT}"
test -f "${MV3_ROOT}/verification.json"
test -f "${REPO_ROOT}/reference_data/mohammadzadeh_2012/mv5_confirmatory_protocol.json"
test -f "${REPO_ROOT}/reference_data/mohammadzadeh_2012/mv6_four_architecture_screen_protocol.json"

python - "${MV3_ROOT}/verification.json" \
  "${REPO_ROOT}/reference_data/mohammadzadeh_2012/mv5_confirmatory_protocol.json" \
  "${REPO_ROOT}/reference_data/mohammadzadeh_2012/mv6_four_architecture_screen_protocol.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

verification = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
mv5_protocol = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
mv6_protocol = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
source = mv6_protocol["source_contract"]
if verification.get("status") != source["mv3_required_verification_status"]:
    raise SystemExit("MV6 requires a verified MV3 source")
if verification.get("summary_sha256") != source["mv3_summary_sha256"]:
    raise SystemExit("MV6 MV3 summary hash mismatch")
if sha(sys.argv[2]) != source["mv5_protocol_sha256"]:
    raise SystemExit("MV6 MV5 protocol hash mismatch")
if mv5_protocol["source_contract"]["mv3_protocol_sha256"] != source["mv3_protocol_sha256"]:
    raise SystemExit("MV6/MV5 MV3 protocol ancestry mismatch")
PY

if [[ -e "${OUTPUT_ROOT}/summary.json" ]]; then
  echo "Refusing to overwrite completed MV6 output: ${OUTPUT_ROOT}" >&2
  exit 2
fi

mkdir -p logs "${OUTPUT_ROOT}"
VENV_DIR="${MV6_VENV_DIR:-}"
REFERENCE_JOB_ID="${MV6_AFTER_REFERENCE_JOB_ID:-}"
if [[ -f "${REPO_ROOT}/LAST_MOHAMMADZADEH_VISION_MV5_JOB.env" ]]; then
  source "${REPO_ROOT}/LAST_MOHAMMADZADEH_VISION_MV5_JOB.env"
  VENV_DIR="${VENV_DIR:-${MV5_VENV_DIR:-}}"
  REFERENCE_JOB_ID="${REFERENCE_JOB_ID:-${MV5_REFERENCE_JOB_ID:-}}"
  MV5_REFERENCE_ROOT="${MV5_OUTPUT_ROOT:-${MV5_REFERENCE_ROOT}}"
fi
if [[ -z "${VENV_DIR}" ]]; then
  for candidate in "${REPO_ROOT}/.venv-mv5" "${REPO_ROOT}/.venv-mv4" "${REPO_ROOT}/.venv-mv3"; do
    if [[ -x "${candidate}/bin/python" ]] && \
       "${candidate}/bin/python" -c 'import numpy, torch, matplotlib' 2>/dev/null; then
      VENV_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${VENV_DIR}" ]]; then
  VENV_DIR="${REPO_ROOT}/.venv-mv6"
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  fi
  source "${VENV_DIR}/bin/activate"
  python -m pip install -e '.[ml,plot]'
else
  source "${VENV_DIR}/bin/activate"
  python -m pip install -e . --no-deps
fi

python -m compileall -q vgdsmc/mohammadzadeh_architecture_screen.py
if python -c 'import pytest' 2>/dev/null; then
  python -m pytest -q tests/test_mohammadzadeh_architecture_screen.py
fi
python - <<'PY'
from vgdsmc import mohammadzadeh_architecture_screen as screen
screen.locked_protocol()
report = screen.parameter_report(7)
if not report["pass"]:
    raise SystemExit(f"parameter parity failed: {report}")
print(report)
PY

if [[ -z "${REFERENCE_JOB_ID}" ]]; then
  completed="$(find "${MV5_REFERENCE_ROOT}/references" -path '*/seed_*/summary.json' -type f 2>/dev/null | wc -l)"
  if [[ "${completed}" -ne 16 ]]; then
    echo "MV6 needs MV5 reference dependency or all 16 completed references" >&2
    exit 3
  fi
fi

EXPORTS="ALL,MV6_REPO_ROOT=${REPO_ROOT},MV6_M3_ROOT=${M3_ROOT},MV6_MV3_ROOT=${MV3_ROOT},MV6_MV5_REFERENCE_ROOT=${MV5_REFERENCE_ROOT},MV6_OUTPUT_ROOT=${OUTPUT_ROOT},MV6_VENV_DIR=${VENV_DIR},MV6_EPOCHS=${MV6_EPOCHS:-200},MV6_BATCH_SIZE=${MV6_BATCH_SIZE:-6}"
DEPENDENCY=()
if [[ -n "${REFERENCE_JOB_ID}" ]]; then
  DEPENDENCY=(--dependency="afterok:${REFERENCE_JOB_ID}")
fi
MODEL_JOB_ID="$(sbatch --parsable "${DEPENDENCY[@]}" --export="${EXPORTS}" scripts/unity_mohammadzadeh_architecture_screen_task.sbatch)"
POST_JOB_ID="$(sbatch --parsable --dependency="afterok:${MODEL_JOB_ID}" --export="${EXPORTS}" scripts/unity_mohammadzadeh_architecture_screen_post.sbatch)"

ENV_FILE="${REPO_ROOT}/LAST_MOHAMMADZADEH_ARCHITECTURE_SCREEN_JOB.env"
printf 'MV6_MODEL_JOB_ID=%q\nMV6_POST_JOB_ID=%q\nMV6_REFERENCE_JOB_ID=%q\nMV6_REPO_ROOT=%q\nMV6_M3_ROOT=%q\nMV6_MV3_ROOT=%q\nMV6_MV5_REFERENCE_ROOT=%q\nMV6_OUTPUT_ROOT=%q\nMV6_VENV_DIR=%q\n' \
  "${MODEL_JOB_ID}" "${POST_JOB_ID}" "${REFERENCE_JOB_ID}" "${REPO_ROOT}" "${M3_ROOT}" "${MV3_ROOT}" "${MV5_REFERENCE_ROOT}" "${OUTPUT_ROOT}" "${VENV_DIR}" > "${ENV_FILE}"

echo "Submitted MV6 four-architecture screen: ${MODEL_JOB_ID} (12 tasks, at most 4 concurrent)"
echo "Submitted MV6 postprocessor: ${POST_JOB_ID}"
if [[ -n "${REFERENCE_JOB_ID}" ]]; then
  echo "MV6 waits for MV5 reference job: ${REFERENCE_JOB_ID}"
fi
echo "Saved: ${ENV_FILE}"
echo "Monitor: squeue -j ${MODEL_JOB_ID},${POST_JOB_ID}"
