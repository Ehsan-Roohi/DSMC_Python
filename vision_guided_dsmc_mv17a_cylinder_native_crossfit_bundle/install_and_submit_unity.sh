#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_TARGET="/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc"
TARGET_ROOT="${1:-${MV17A_PROJECT_ROOT:-${DEFAULT_TARGET}}}"
PAYLOAD_TARGET="${TARGET_ROOT}/mv17a_cylinder_native_crossfit"

if [[ ! -d "${TARGET_ROOT}" || ! -f "${TARGET_ROOT}/vgdsmc/mohammadzadeh_architecture_screen.py" ]]; then
  echo "MV17A_INSTALL_ERROR: target is not the canonical vision_guided_dsmc project: ${TARGET_ROOT}" >&2
  exit 2
fi
test -s "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV16B_JCP_EVIDENCE_RESULT.env"
test -s "${TARGET_ROOT}/vgdsmc/mohammadzadeh_mv16b_jcp_evidence_audit.py"

for relative in \
  vgdsmc/mohammadzadeh_mv17a_cylinder_native_crossfit.py \
  tests/test_mohammadzadeh_mv17a_cylinder_native_crossfit.py \
  reference_data/mohammadzadeh_2012/mv17a_cylinder_native_crossfit_protocol.json \
  scripts/submit_mohammadzadeh_mv17a_unity.sh \
  scripts/unity_mohammadzadeh_mv17a_audit.sbatch \
  scripts/unity_mohammadzadeh_mv17a_post.sbatch; do
  install -D -m 0644 "${BUNDLE_ROOT}/payload/${relative}" "${PAYLOAD_TARGET}/${relative}"
done
install -D -m 0644 \
  "${BUNDLE_ROOT}/payload/vgdsmc/mohammadzadeh_mv17a_cylinder_native_crossfit.py" \
  "${TARGET_ROOT}/vgdsmc/mohammadzadeh_mv17a_cylinder_native_crossfit.py"
install -D -m 0644 \
  "${BUNDLE_ROOT}/payload/reference_data/mohammadzadeh_2012/mv17a_cylinder_native_crossfit_protocol.json" \
  "${TARGET_ROOT}/reference_data/mohammadzadeh_2012/mv17a_cylinder_native_crossfit_protocol.json"
chmod +x \
  "${PAYLOAD_TARGET}/scripts/submit_mohammadzadeh_mv17a_unity.sh" \
  "${PAYLOAD_TARGET}/scripts/unity_mohammadzadeh_mv17a_audit.sbatch" \
  "${PAYLOAD_TARGET}/scripts/unity_mohammadzadeh_mv17a_post.sbatch"

source "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV16B_JCP_EVIDENCE_JOB.env"
PYTHON_BIN="${MV17A_PYTHON:-${MV16B_PYTHON:-}}"
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "MV17A_INSTALL_ERROR: no usable MV17A/MV16B Python was found" >&2
  exit 3
fi
if ! "${PYTHON_BIN}" -c 'import matplotlib,numpy,scipy' >/dev/null 2>&1; then
  echo "MV17A_INSTALL_ERROR: selected Python lacks matplotlib, numpy, or scipy: ${PYTHON_BIN}" >&2
  exit 4
fi

cd "${TARGET_ROOT}"
"${PYTHON_BIN}" -m py_compile vgdsmc/mohammadzadeh_mv17a_cylinder_native_crossfit.py
MPLCONFIGDIR="${TMPDIR:-/tmp}/mv17a-install-mpl" \
  "${PYTHON_BIN}" "${PAYLOAD_TARGET}/tests/test_mohammadzadeh_mv17a_cylinder_native_crossfit.py"
MV17A_PROJECT_ROOT="${TARGET_ROOT}" MV17A_PYTHON="${PYTHON_BIN}" \
  exec bash "${PAYLOAD_TARGET}/scripts/submit_mohammadzadeh_mv17a_unity.sh"
