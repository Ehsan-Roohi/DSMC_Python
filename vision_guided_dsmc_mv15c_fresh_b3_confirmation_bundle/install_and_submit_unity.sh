#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MV9_BUNDLE_ROOT="$(cd "${BUNDLE_ROOT}/../vision_guided_dsmc_mv9_heat_flux_bundle" && pwd)"
MV14_BUNDLE_ROOT="$(cd "${BUNDLE_ROOT}/../vision_guided_dsmc_mv14_kinetic_conservation_cavity_bundle" && pwd)"
MV15A_BUNDLE_ROOT="$(cd "${BUNDLE_ROOT}/../vision_guided_dsmc_mv15a_spectral_information_audit_bundle" && pwd)"
MV15B_BUNDLE_ROOT="$(cd "${BUNDLE_ROOT}/../vision_guided_dsmc_mv15b_data_consistent_budget_bundle" && pwd)"
DEFAULT_TARGET="/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc"
TARGET_ROOT="${1:-${MV15C_PROJECT_ROOT:-${DEFAULT_TARGET}}}"
PAYLOAD_TARGET="${TARGET_ROOT}/mv15c_fresh_b3_confirmation"

if [[ ! -d "${TARGET_ROOT}" || ! -f "${TARGET_ROOT}/vgdsmc/mohammadzadeh_architecture_screen.py" ]]; then
  echo "MV15C_INSTALL_ERROR: target is not the canonical vision_guided_dsmc project: ${TARGET_ROOT}" >&2
  exit 2
fi
for relative in \
  vgdsmc/mohammadzadeh_mv3_reference.py \
  vgdsmc/mohammadzadeh_vision_mv3.py \
  vgdsmc/mohammadzadeh_spatial_refinement.py \
  reference_data/mohammadzadeh_2012/mv3_cross_condition_protocol.json \
  reference_data/mohammadzadeh_2012/m1r_repair_protocol.json; do
  test -s "${TARGET_ROOT}/${relative}"
done
test -s "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV10_QY_JOB.env"
test -s "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV15B_DATA_CONSISTENT_BUDGET_JOB.env"
test -s "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV15B_DATA_CONSISTENT_BUDGET_RESULT.env"

for relative in \
  vgdsmc/mohammadzadeh_mv9_heat_flux.py \
  reference_data/mohammadzadeh_2012/mv9_heat_flux_noise2noise_protocol.json; do
  install -D -m 0644 "${MV9_BUNDLE_ROOT}/payload/${relative}" "${TARGET_ROOT}/${relative}"
done
for relative in \
  vgdsmc/mohammadzadeh_mv14_kinetic_conservation_cavity.py \
  reference_data/mohammadzadeh_2012/mv14_kinetic_conservation_cavity_protocol.json; do
  install -D -m 0644 "${MV14_BUNDLE_ROOT}/payload/${relative}" "${TARGET_ROOT}/${relative}"
done
for relative in \
  vgdsmc/mohammadzadeh_mv15a_spectral_information_audit.py \
  reference_data/mohammadzadeh_2012/mv15a_spectral_information_audit_protocol.json; do
  install -D -m 0644 "${MV15A_BUNDLE_ROOT}/payload/${relative}" "${TARGET_ROOT}/${relative}"
done
for relative in \
  vgdsmc/mohammadzadeh_mv15b_data_consistent_budget.py \
  reference_data/mohammadzadeh_2012/mv15b_data_consistent_budget_protocol.json; do
  install -D -m 0644 "${MV15B_BUNDLE_ROOT}/payload/${relative}" "${TARGET_ROOT}/${relative}"
done
for relative in \
  vgdsmc/mohammadzadeh_mv15c_fresh_b3_confirmation.py \
  tests/test_mohammadzadeh_mv15c_fresh_b3_confirmation.py \
  reference_data/mohammadzadeh_2012/mv15c_fresh_b3_confirmation_protocol.json \
  scripts/submit_mohammadzadeh_mv15c_fresh_b3_unity.sh \
  scripts/unity_mohammadzadeh_mv15c_reference.sbatch \
  scripts/unity_mohammadzadeh_mv15c_predict.sbatch \
  scripts/unity_mohammadzadeh_mv15c_post.sbatch; do
  install -D -m 0644 "${BUNDLE_ROOT}/payload/${relative}" "${PAYLOAD_TARGET}/${relative}"
done
install -D -m 0644 \
  "${BUNDLE_ROOT}/payload/vgdsmc/mohammadzadeh_mv15c_fresh_b3_confirmation.py" \
  "${TARGET_ROOT}/vgdsmc/mohammadzadeh_mv15c_fresh_b3_confirmation.py"
install -D -m 0644 \
  "${BUNDLE_ROOT}/payload/reference_data/mohammadzadeh_2012/mv15c_fresh_b3_confirmation_protocol.json" \
  "${TARGET_ROOT}/reference_data/mohammadzadeh_2012/mv15c_fresh_b3_confirmation_protocol.json"
chmod +x \
  "${PAYLOAD_TARGET}/scripts/submit_mohammadzadeh_mv15c_fresh_b3_unity.sh" \
  "${PAYLOAD_TARGET}/scripts/unity_mohammadzadeh_mv15c_reference.sbatch" \
  "${PAYLOAD_TARGET}/scripts/unity_mohammadzadeh_mv15c_predict.sbatch" \
  "${PAYLOAD_TARGET}/scripts/unity_mohammadzadeh_mv15c_post.sbatch"

source "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV10_QY_JOB.env"
test -x "${MV10_VENV_DIR:?}/bin/python"
PYTHON_BIN="${MV15C_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  if "${CONDA_PREFIX}/bin/python" -c 'import matplotlib, numpy, torch' >/dev/null 2>&1; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  fi
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${MV10_VENV_DIR}/bin/python"
fi
if [[ ! -x "${PYTHON_BIN}" ]] || ! "${PYTHON_BIN}" -c 'import matplotlib, numpy, torch' >/dev/null 2>&1; then
  echo "MV15C_INSTALL_ERROR: selected Python lacks matplotlib, numpy, or torch: ${PYTHON_BIN}" >&2
  echo "Activate dsmc-gpu or set MV15C_PYTHON=/absolute/path/to/python." >&2
  exit 4
fi

cd "${TARGET_ROOT}"
"${PYTHON_BIN}" -m py_compile vgdsmc/mohammadzadeh_mv15c_fresh_b3_confirmation.py
PYTHONPATH="${TARGET_ROOT}:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" "${PAYLOAD_TARGET}/tests/test_mohammadzadeh_mv15c_fresh_b3_confirmation.py"
MV15C_PROJECT_ROOT="${TARGET_ROOT}" MV15C_PYTHON="${PYTHON_BIN}" \
  exec bash "${PAYLOAD_TARGET}/scripts/submit_mohammadzadeh_mv15c_fresh_b3_unity.sh"
