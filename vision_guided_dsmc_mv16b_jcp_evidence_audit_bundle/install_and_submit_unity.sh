#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_TARGET="/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc"
TARGET_ROOT="${1:-${MV16B_PROJECT_ROOT:-${DEFAULT_TARGET}}}"
PAYLOAD_TARGET="${TARGET_ROOT}/mv16b_jcp_evidence_audit"

if [[ ! -d "${TARGET_ROOT}" || ! -f "${TARGET_ROOT}/vgdsmc/mohammadzadeh_architecture_screen.py" ]]; then
  echo "MV16B_INSTALL_ERROR: target is not the canonical vision_guided_dsmc project: ${TARGET_ROOT}" >&2
  exit 2
fi
for required in \
  LAST_MV11_DS2V_CYLINDER_RESULT.env \
  LAST_MOHAMMADZADEH_MV15C_A1_QY_JOB.env \
  LAST_MOHAMMADZADEH_MV15C_A1_QY_RESULT.env \
  LAST_MOHAMMADZADEH_MV16A_CYLINDER_JOB.env; do
  test -s "${TARGET_ROOT}/${required}"
done
for required in \
  vgdsmc/mohammadzadeh_mv9_heat_flux.py \
  vgdsmc/mohammadzadeh_mv14_kinetic_conservation_cavity.py \
  vgdsmc/mohammadzadeh_mv15a_spectral_information_audit.py \
  vgdsmc/mohammadzadeh_mv15b_data_consistent_budget.py \
  vgdsmc/mohammadzadeh_mv15c_fresh_b3_confirmation.py \
  vgdsmc/mohammadzadeh_mv16a_frozen_cylinder_transfer.py; do
  test -s "${TARGET_ROOT}/${required}"
done

for relative in \
  vgdsmc/mohammadzadeh_mv16b_jcp_evidence_audit.py \
  tests/test_mohammadzadeh_mv16b_jcp_evidence_audit.py \
  reference_data/mohammadzadeh_2012/mv16b_jcp_evidence_audit_protocol.json \
  scripts/submit_mohammadzadeh_mv16b_unity.sh \
  scripts/unity_mohammadzadeh_mv16b_audit.sbatch \
  scripts/unity_mohammadzadeh_mv16b_post.sbatch; do
  install -D -m 0644 "${BUNDLE_ROOT}/payload/${relative}" "${PAYLOAD_TARGET}/${relative}"
done
install -D -m 0644 \
  "${BUNDLE_ROOT}/payload/vgdsmc/mohammadzadeh_mv16b_jcp_evidence_audit.py" \
  "${TARGET_ROOT}/vgdsmc/mohammadzadeh_mv16b_jcp_evidence_audit.py"
install -D -m 0644 \
  "${BUNDLE_ROOT}/payload/reference_data/mohammadzadeh_2012/mv16b_jcp_evidence_audit_protocol.json" \
  "${TARGET_ROOT}/reference_data/mohammadzadeh_2012/mv16b_jcp_evidence_audit_protocol.json"
chmod +x \
  "${PAYLOAD_TARGET}/scripts/submit_mohammadzadeh_mv16b_unity.sh" \
  "${PAYLOAD_TARGET}/scripts/unity_mohammadzadeh_mv16b_audit.sbatch" \
  "${PAYLOAD_TARGET}/scripts/unity_mohammadzadeh_mv16b_post.sbatch"

source "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV15C_A1_QY_JOB.env"
PYTHON_BIN="${MV16B_PYTHON:-${MV15C_A1_PYTHON:-}}"
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "MV16B_INSTALL_ERROR: no usable MV16B/MV15C-A1 Python was found" >&2
  exit 3
fi
if ! "${PYTHON_BIN}" -c 'import matplotlib,numpy,scipy,torch' >/dev/null 2>&1; then
  echo "MV16B_INSTALL_ERROR: selected Python lacks matplotlib, numpy, scipy, or torch: ${PYTHON_BIN}" >&2
  exit 4
fi

cd "${TARGET_ROOT}"
"${PYTHON_BIN}" -m py_compile vgdsmc/mohammadzadeh_mv16b_jcp_evidence_audit.py
MPLCONFIGDIR="${TMPDIR:-/tmp}/mv16b-install-mpl" \
  "${PYTHON_BIN}" "${PAYLOAD_TARGET}/tests/test_mohammadzadeh_mv16b_jcp_evidence_audit.py"
MV16B_PROJECT_ROOT="${TARGET_ROOT}" MV16B_PYTHON="${PYTHON_BIN}" \
  exec bash "${PAYLOAD_TARGET}/scripts/submit_mohammadzadeh_mv16b_unity.sh"

