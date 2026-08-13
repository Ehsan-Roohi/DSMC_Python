#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MV8_BUNDLE_ROOT="$(cd "${BUNDLE_ROOT}/../vision_guided_dsmc_mv8_kinetic_moments_bundle" && pwd)"
MV9_BUNDLE_ROOT="$(cd "${BUNDLE_ROOT}/../vision_guided_dsmc_mv9_heat_flux_bundle" && pwd)"
DEFAULT_TARGET="/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc"
TARGET_ROOT="${1:-${MV10_TARGET_ROOT:-${DEFAULT_TARGET}}}"

test -d "${TARGET_ROOT}"
test -f "${TARGET_ROOT}/vgdsmc/mohammadzadeh_architecture_screen.py"
test -f "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV9_HEAT_FLUX_JOB.env"

# Reinstall the exact immutable ancestors required by the MV10 hash lock.
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
  tests/test_mohammadzadeh_mv10_qy_multiscale.py \
  reference_data/mohammadzadeh_2012/mv10_qy_multiscale_bias_repair_protocol.json \
  scripts/submit_mohammadzadeh_mv10_qy_unity.sh \
  scripts/unity_mohammadzadeh_mv10_qy_assemble.sbatch \
  scripts/unity_mohammadzadeh_mv10_qy_model.sbatch \
  scripts/unity_mohammadzadeh_mv10_qy_post.sbatch; do
  install -D -m 0644 "${BUNDLE_ROOT}/payload/${relative}" "${TARGET_ROOT}/${relative}"
done

chmod +x \
  "${TARGET_ROOT}/scripts/submit_mohammadzadeh_mv10_qy_unity.sh" \
  "${TARGET_ROOT}/scripts/unity_mohammadzadeh_mv10_qy_assemble.sbatch" \
  "${TARGET_ROOT}/scripts/unity_mohammadzadeh_mv10_qy_model.sbatch" \
  "${TARGET_ROOT}/scripts/unity_mohammadzadeh_mv10_qy_post.sbatch"

cd "${TARGET_ROOT}"
python3 -m py_compile vgdsmc/mohammadzadeh_mv10_qy_multiscale.py
exec bash scripts/submit_mohammadzadeh_mv10_qy_unity.sh
