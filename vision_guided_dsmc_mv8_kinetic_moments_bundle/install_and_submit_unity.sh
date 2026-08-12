#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_TARGET="/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc"
TARGET_ROOT="${1:-${MV8_TARGET_ROOT:-${DEFAULT_TARGET}}}"

test -d "${TARGET_ROOT}"
test -f "${TARGET_ROOT}/vgdsmc/mohammadzadeh_architecture_screen.py"
test -f "${TARGET_ROOT}/vgdsmc/mohammadzadeh_mv7_jcp_budget_matrix.py"
test -f "${TARGET_ROOT}/vgdsmc/ntc_checkpoint.py"
test -f "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV7_JCP_JOB.env"

for relative in \
  vgdsmc/mohammadzadeh_mv8_kinetic_moments.py \
  tests/test_mohammadzadeh_mv8_kinetic_moments.py \
  reference_data/mohammadzadeh_2012/mv8_kinetic_moment_feasibility_protocol.json \
  scripts/submit_mohammadzadeh_mv8_kinetic_unity.sh \
  scripts/unity_mohammadzadeh_mv8_assemble.sbatch \
  scripts/unity_mohammadzadeh_mv8_model.sbatch \
  scripts/unity_mohammadzadeh_mv8_post.sbatch; do
  install -D -m 0644 "${BUNDLE_ROOT}/payload/${relative}" "${TARGET_ROOT}/${relative}"
done

chmod +x \
  "${TARGET_ROOT}/scripts/submit_mohammadzadeh_mv8_kinetic_unity.sh" \
  "${TARGET_ROOT}/scripts/unity_mohammadzadeh_mv8_assemble.sbatch" \
  "${TARGET_ROOT}/scripts/unity_mohammadzadeh_mv8_model.sbatch" \
  "${TARGET_ROOT}/scripts/unity_mohammadzadeh_mv8_post.sbatch"

cd "${TARGET_ROOT}"
python3 -m py_compile vgdsmc/mohammadzadeh_mv8_kinetic_moments.py
exec bash scripts/submit_mohammadzadeh_mv8_kinetic_unity.sh
