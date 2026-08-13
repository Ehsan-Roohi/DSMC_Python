#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MV8_BUNDLE_ROOT="$(cd "${BUNDLE_ROOT}/../vision_guided_dsmc_mv8_kinetic_moments_bundle" && pwd)"
MV9_BUNDLE_ROOT="$(cd "${BUNDLE_ROOT}/../vision_guided_dsmc_mv9_heat_flux_bundle" && pwd)"
MV10_BUNDLE_ROOT="$(cd "${BUNDLE_ROOT}/../vision_guided_dsmc_mv10_qy_multiscale_bundle" && pwd)"
DEFAULT_TARGET="/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc"
TARGET_ROOT="${1:-${MV12_PROJECT_ROOT:-${DEFAULT_TARGET}}}"
PAYLOAD_TARGET="${TARGET_ROOT}/mv12_sage_qy"

test -d "${TARGET_ROOT}"
test -f "${TARGET_ROOT}/vgdsmc/mohammadzadeh_architecture_screen.py"
test -s "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV10_QY_JOB.env"

# Reinstall the exact immutable ancestors required by the MV12 source lock.
for relative in \
  vgdsmc/mohammadzadeh_mv8_kinetic_moments.py \
  reference_data/mohammadzadeh_2012/mv8_kinetic_moment_feasibility_protocol.json; do
  install -D -m 0644 "${MV8_BUNDLE_ROOT}/payload/${relative}" "${TARGET_ROOT}/${relative}"
done

for relative in \
  vgdsmc/mohammadzadeh_mv9_heat_flux.py \
  reference_data/mohammadzadeh_2012/mv9_heat_flux_noise2noise_protocol.json; do
  install -D -m 0644 "${MV9_BUNDLE_ROOT}/payload/${relative}" "${TARGET_ROOT}/${relative}"
done

for relative in \
  vgdsmc/mohammadzadeh_mv10_qy_multiscale.py \
  reference_data/mohammadzadeh_2012/mv10_qy_multiscale_bias_repair_protocol.json; do
  install -D -m 0644 "${MV10_BUNDLE_ROOT}/payload/${relative}" "${TARGET_ROOT}/${relative}"
done

for relative in \
  vgdsmc/mohammadzadeh_mv12_sage_qy.py \
  tests/test_mohammadzadeh_mv12_sage_qy.py \
  reference_data/mohammadzadeh_2012/mv12_sage_qy_protocol.json \
  scripts/submit_mohammadzadeh_mv12_sage_unity.sh \
  scripts/unity_mohammadzadeh_mv12_sage_predict.sbatch \
  scripts/unity_mohammadzadeh_mv12_sage_post.sbatch; do
  install -D -m 0644 "${BUNDLE_ROOT}/payload/${relative}" "${PAYLOAD_TARGET}/${relative}"
done

# The module/protocol live in the project package; jobs/tests live in an isolated payload.
install -D -m 0644 \
  "${BUNDLE_ROOT}/payload/vgdsmc/mohammadzadeh_mv12_sage_qy.py" \
  "${TARGET_ROOT}/vgdsmc/mohammadzadeh_mv12_sage_qy.py"
install -D -m 0644 \
  "${BUNDLE_ROOT}/payload/reference_data/mohammadzadeh_2012/mv12_sage_qy_protocol.json" \
  "${TARGET_ROOT}/reference_data/mohammadzadeh_2012/mv12_sage_qy_protocol.json"

chmod +x \
  "${PAYLOAD_TARGET}/scripts/submit_mohammadzadeh_mv12_sage_unity.sh" \
  "${PAYLOAD_TARGET}/scripts/unity_mohammadzadeh_mv12_sage_predict.sbatch" \
  "${PAYLOAD_TARGET}/scripts/unity_mohammadzadeh_mv12_sage_post.sbatch"

cd "${TARGET_ROOT}"
python3 -m py_compile vgdsmc/mohammadzadeh_mv12_sage_qy.py
PYTHONPATH="${TARGET_ROOT}:${PYTHONPATH:-}" \
  python3 -m pytest -q "${PAYLOAD_TARGET}/tests/test_mohammadzadeh_mv12_sage_qy.py"
MV12_PROJECT_ROOT="${TARGET_ROOT}" \
  exec bash "${PAYLOAD_TARGET}/scripts/submit_mohammadzadeh_mv12_sage_unity.sh"
