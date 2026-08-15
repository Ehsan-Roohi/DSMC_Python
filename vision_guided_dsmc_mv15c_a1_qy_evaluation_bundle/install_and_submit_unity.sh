#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MV15C_BUNDLE_ROOT="$(cd "${BUNDLE_ROOT}/../vision_guided_dsmc_mv15c_fresh_b3_confirmation_bundle" && pwd)"
DEFAULT_TARGET="/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc"
TARGET_ROOT="${1:-${MV15C_A1_PROJECT_ROOT:-${DEFAULT_TARGET}}}"
PAYLOAD_TARGET="${TARGET_ROOT}/mv15c_a1_qy_evaluation"

if [[ ! -d "${TARGET_ROOT}" || ! -f "${TARGET_ROOT}/vgdsmc/mohammadzadeh_architecture_screen.py" ]]; then
  echo "MV15C_A1_INSTALL_ERROR: target is not the canonical vision_guided_dsmc project: ${TARGET_ROOT}" >&2
  exit 2
fi
test -s "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV15C_FRESH_B3_JOB.env"
test -s "${MV15C_BUNDLE_ROOT}/payload/vgdsmc/mohammadzadeh_mv15c_fresh_b3_confirmation.py"
test -s "${MV15C_BUNDLE_ROOT}/payload/reference_data/mohammadzadeh_2012/mv15c_fresh_b3_confirmation_protocol.json"

# Restore the exact reviewed MV15C implementation that generated the eight
# references; do not call its installer because that would submit new DSMC.
install -D -m 0644 \
  "${MV15C_BUNDLE_ROOT}/payload/vgdsmc/mohammadzadeh_mv15c_fresh_b3_confirmation.py" \
  "${TARGET_ROOT}/vgdsmc/mohammadzadeh_mv15c_fresh_b3_confirmation.py"
install -D -m 0644 \
  "${MV15C_BUNDLE_ROOT}/payload/reference_data/mohammadzadeh_2012/mv15c_fresh_b3_confirmation_protocol.json" \
  "${TARGET_ROOT}/reference_data/mohammadzadeh_2012/mv15c_fresh_b3_confirmation_protocol.json"

for relative in \
  vgdsmc/mohammadzadeh_mv15c_a1_qy_evaluation.py \
  tests/test_mohammadzadeh_mv15c_a1_qy_evaluation.py \
  reference_data/mohammadzadeh_2012/mv15c_a1_qy_evaluation_amendment.json \
  scripts/submit_mohammadzadeh_mv15c_a1_qy_unity.sh \
  scripts/unity_mohammadzadeh_mv15c_a1_predict.sbatch \
  scripts/unity_mohammadzadeh_mv15c_a1_post.sbatch; do
  install -D -m 0644 "${BUNDLE_ROOT}/payload/${relative}" "${PAYLOAD_TARGET}/${relative}"
done
install -D -m 0644 \
  "${BUNDLE_ROOT}/payload/vgdsmc/mohammadzadeh_mv15c_a1_qy_evaluation.py" \
  "${TARGET_ROOT}/vgdsmc/mohammadzadeh_mv15c_a1_qy_evaluation.py"
install -D -m 0644 \
  "${BUNDLE_ROOT}/payload/reference_data/mohammadzadeh_2012/mv15c_a1_qy_evaluation_amendment.json" \
  "${TARGET_ROOT}/reference_data/mohammadzadeh_2012/mv15c_a1_qy_evaluation_amendment.json"
chmod +x \
  "${PAYLOAD_TARGET}/scripts/submit_mohammadzadeh_mv15c_a1_qy_unity.sh" \
  "${PAYLOAD_TARGET}/scripts/unity_mohammadzadeh_mv15c_a1_predict.sbatch" \
  "${PAYLOAD_TARGET}/scripts/unity_mohammadzadeh_mv15c_a1_post.sbatch"

source "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV15C_FRESH_B3_JOB.env"
PYTHON_BIN="${MV15C_A1_PYTHON:-${MV15C_PYTHON:-}}"
if [[ -z "${PYTHON_BIN}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "MV15C_A1_INSTALL_ERROR: no usable Python was found in the MV15C pointer or environment." >&2
  exit 4
fi
if ! "${PYTHON_BIN}" -c 'import matplotlib, numpy, torch' >/dev/null 2>&1; then
  echo "MV15C_A1_INSTALL_ERROR: selected Python lacks matplotlib, numpy, or torch: ${PYTHON_BIN}" >&2
  exit 5
fi

cd "${TARGET_ROOT}"
"${PYTHON_BIN}" -m py_compile \
  vgdsmc/mohammadzadeh_mv15c_fresh_b3_confirmation.py \
  vgdsmc/mohammadzadeh_mv15c_a1_qy_evaluation.py
PYTHONPATH="${TARGET_ROOT}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" "${MV15C_BUNDLE_ROOT}/payload/tests/test_mohammadzadeh_mv15c_fresh_b3_confirmation.py"
PYTHONPATH="${TARGET_ROOT}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" "${PAYLOAD_TARGET}/tests/test_mohammadzadeh_mv15c_a1_qy_evaluation.py"
MV15C_A1_PROJECT_ROOT="${TARGET_ROOT}" MV15C_A1_PYTHON="${PYTHON_BIN}" \
  exec bash "${PAYLOAD_TARGET}/scripts/submit_mohammadzadeh_mv15c_a1_qy_unity.sh"

