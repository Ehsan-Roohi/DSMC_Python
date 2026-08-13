#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MV8_BUNDLE_ROOT="$(cd "${BUNDLE_ROOT}/../vision_guided_dsmc_mv8_kinetic_moments_bundle" && pwd)"
DEFAULT_TARGET="/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc"
TARGET_ROOT="${1:-${MV9_TARGET_ROOT:-${DEFAULT_TARGET}}}"

test -d "${TARGET_ROOT}"
test -f "${TARGET_ROOT}/vgdsmc/mohammadzadeh_architecture_screen.py"
test -f "${TARGET_ROOT}/vgdsmc/mohammadzadeh_mv7_jcp_budget_matrix.py"
test -f "${TARGET_ROOT}/vgdsmc/ntc_checkpoint.py"
test -f "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV7_JCP_JOB.env"

# Install the exact immutable MV8 source ancestor used by the MV9 hash lock.
# This does not execute or alter any MV8 result; it only makes a fresh target
# reproducible and prevents dependence on whichever local source copy survived.
for relative in \
  vgdsmc/mohammadzadeh_mv8_kinetic_moments.py \
  reference_data/mohammadzadeh_2012/mv8_kinetic_moment_feasibility_protocol.json; do
  install -D -m 0644 "${MV8_BUNDLE_ROOT}/payload/${relative}" "${TARGET_ROOT}/${relative}"
done

for relative in \
  vgdsmc/mohammadzadeh_mv9_heat_flux.py \
  tests/test_mohammadzadeh_mv9_heat_flux.py \
  reference_data/mohammadzadeh_2012/mv9_heat_flux_noise2noise_protocol.json \
  scripts/submit_mohammadzadeh_mv9_heat_flux_unity.sh \
  scripts/unity_mohammadzadeh_mv9_assemble.sbatch \
  scripts/unity_mohammadzadeh_mv9_model.sbatch \
  scripts/unity_mohammadzadeh_mv9_post.sbatch; do
  install -D -m 0644 "${BUNDLE_ROOT}/payload/${relative}" "${TARGET_ROOT}/${relative}"
done

chmod +x \
  "${TARGET_ROOT}/scripts/submit_mohammadzadeh_mv9_heat_flux_unity.sh" \
  "${TARGET_ROOT}/scripts/unity_mohammadzadeh_mv9_assemble.sbatch" \
  "${TARGET_ROOT}/scripts/unity_mohammadzadeh_mv9_model.sbatch" \
  "${TARGET_ROOT}/scripts/unity_mohammadzadeh_mv9_post.sbatch"

cd "${TARGET_ROOT}"
python3 -m py_compile vgdsmc/mohammadzadeh_mv9_heat_flux.py
exec bash scripts/submit_mohammadzadeh_mv9_heat_flux_unity.sh
