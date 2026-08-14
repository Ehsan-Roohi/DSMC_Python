#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MV9_BUNDLE_ROOT="$(cd "${BUNDLE_ROOT}/../vision_guided_dsmc_mv9_heat_flux_bundle" && pwd)"
MV14_BUNDLE_ROOT="$(cd "${BUNDLE_ROOT}/../vision_guided_dsmc_mv14_kinetic_conservation_cavity_bundle" && pwd)"
DEFAULT_TARGET="/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc"
TARGET_ROOT="${1:-${MV15A_PROJECT_ROOT:-${DEFAULT_TARGET}}}"
PAYLOAD_TARGET="${TARGET_ROOT}/mv15a_spectral_information_audit"

if [[ ! -d "${TARGET_ROOT}" || ! -f "${TARGET_ROOT}/vgdsmc/mohammadzadeh_architecture_screen.py" ]]; then
  echo "MV15A_INSTALL_ERROR: target is not the canonical vision_guided_dsmc project: ${TARGET_ROOT}" >&2
  exit 2
fi
test -s "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV10_QY_JOB.env"
test -s "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV14_KINETIC_CAVITY_JOB.env"
test -s "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV14_KINETIC_CAVITY_RESULT.env"

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
  tests/test_mohammadzadeh_mv15a_spectral_information_audit.py \
  reference_data/mohammadzadeh_2012/mv15a_spectral_information_audit_protocol.json \
  scripts/submit_mohammadzadeh_mv15a_spectral_audit_unity.sh \
  scripts/unity_mohammadzadeh_mv15a_spectral_predict.sbatch \
  scripts/unity_mohammadzadeh_mv15a_spectral_post.sbatch; do
  install -D -m 0644 "${BUNDLE_ROOT}/payload/${relative}" "${PAYLOAD_TARGET}/${relative}"
done
install -D -m 0644 \
  "${BUNDLE_ROOT}/payload/vgdsmc/mohammadzadeh_mv15a_spectral_information_audit.py" \
  "${TARGET_ROOT}/vgdsmc/mohammadzadeh_mv15a_spectral_information_audit.py"
install -D -m 0644 \
  "${BUNDLE_ROOT}/payload/reference_data/mohammadzadeh_2012/mv15a_spectral_information_audit_protocol.json" \
  "${TARGET_ROOT}/reference_data/mohammadzadeh_2012/mv15a_spectral_information_audit_protocol.json"
chmod +x \
  "${PAYLOAD_TARGET}/scripts/submit_mohammadzadeh_mv15a_spectral_audit_unity.sh" \
  "${PAYLOAD_TARGET}/scripts/unity_mohammadzadeh_mv15a_spectral_predict.sbatch" \
  "${PAYLOAD_TARGET}/scripts/unity_mohammadzadeh_mv15a_spectral_post.sbatch"

source "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV10_QY_JOB.env"
test -x "${MV10_VENV_DIR:?}/bin/python"
cd "${TARGET_ROOT}"
"${MV10_VENV_DIR}/bin/python" -m py_compile \
  vgdsmc/mohammadzadeh_mv15a_spectral_information_audit.py
PYTHONPATH="${TARGET_ROOT}:${PYTHONPATH:-}" \
  "${MV10_VENV_DIR}/bin/python" \
  "${PAYLOAD_TARGET}/tests/test_mohammadzadeh_mv15a_spectral_information_audit.py"
MV15A_PROJECT_ROOT="${TARGET_ROOT}" \
  exec bash "${PAYLOAD_TARGET}/scripts/submit_mohammadzadeh_mv15a_spectral_audit_unity.sh"
