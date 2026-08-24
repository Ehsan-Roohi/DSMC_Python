#!/usr/bin/env bash
set -euo pipefail

ROOT="${JCP_ROOT:-/project/pi_roohie_umass_edu/DSMC_Python_M3_QY}"
OUT="${JCP10_OUTPUT:-${ROOT}/JCP10_SUPPORT_AUDIT}"
REPO="${JCP10_REPO:-${OUT}/code}"
CODE_COMMIT="${JCP10_CODE_COMMIT:-088a568beb58b620e6eeacb5e3491b08ad81175d}"
RAW="https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/${CODE_COMMIT}"

mkdir -p "${OUT}"
export MPLCONFIGDIR="${JCP10_MPLCONFIGDIR:-${OUT}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"
mkdir -p "${REPO}/scripts"
curl --retry 3 -fsSL "${RAW}/scripts/jcp10_support_gate.py" \
  -o "${REPO}/scripts/jcp10_support_gate.py"
python -m py_compile "${REPO}/scripts/jcp10_support_gate.py"

python "${REPO}/scripts/jcp10_support_gate.py" \
  --model-lock "${ROOT}/JCP6R_MODEL_LOCK/JCP6R_MODEL_LOCK.zip" \
  --prediction-lock "${ROOT}/JCP7_M12_EVALUATION/JCP7_M12_PREDICTION_LOCK.zip" \
  --reference "${ROOT}/JCP8_M12_REFERENCE/JCP8_M12_REFERENCE.zip" \
  --output "${OUT}"

cd "${OUT}"
sha256sum -c JCP10_SUPPORT_AUDIT.zip.sha256
echo "JCP10_SUPPORT_AUDIT_COMPLETE=1"
echo "JCP10_CODE_COMMIT=${CODE_COMMIT}"
echo "UPLOAD=${OUT}/JCP10_SUPPORT_AUDIT.zip ${OUT}/JCP10_SUPPORT_AUDIT.zip.sha256"
