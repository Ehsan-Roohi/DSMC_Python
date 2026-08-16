#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_TARGET="/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc"
TARGET_ROOT="${1:-${MV17B_PROJECT_ROOT:-${DEFAULT_TARGET}}}"
PAYLOAD_TARGET="${TARGET_ROOT}/mv17b_a3_benchmark_independent_recovery"
test -d "${TARGET_ROOT}"
test -s "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV17B_FRESH_CYLINDER_JOB.env"
test -s "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV17B_A1_MECHANICAL_RECOVERY_JOB.env"
test -s "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV17B_A2_LOCKED_WINDOW_RECOVERY_JOB.env"
test -s "${TARGET_ROOT}/vgdsmc/mohammadzadeh_mv17b_fresh_cylinder_confirmation.py"
test -s "${TARGET_ROOT}/mv17b_fresh_cylinder_confirmation/scripts/unity_mohammadzadeh_mv17b_post.sbatch"

for relative in \
  vgdsmc/mohammadzadeh_mv17b_a3_benchmark_independent_recovery.py \
  tests/test_mohammadzadeh_mv17b_a3_benchmark_independent_recovery.py \
  reference_data/mohammadzadeh_2012/mv17b_a3_benchmark_independent_recovery_amendment.json \
  scripts/submit_mohammadzadeh_mv17b_a3_unity.sh \
  scripts/unity_mohammadzadeh_mv17b_a3_prepare.sbatch \
  scripts/unity_mohammadzadeh_mv17b_a3_recover_array.sbatch \
  scripts/unity_mohammadzadeh_mv17b_a3_analyze.sbatch \
  scripts/unity_mohammadzadeh_mv17b_a3_post.sbatch; do
  install -D -m 0644 "${BUNDLE_ROOT}/payload/${relative}" "${PAYLOAD_TARGET}/${relative}"
done
install -D -m 0644 \
  "${BUNDLE_ROOT}/payload/vgdsmc/mohammadzadeh_mv17b_a3_benchmark_independent_recovery.py" \
  "${TARGET_ROOT}/vgdsmc/mohammadzadeh_mv17b_a3_benchmark_independent_recovery.py"
install -D -m 0644 \
  "${BUNDLE_ROOT}/payload/reference_data/mohammadzadeh_2012/mv17b_a3_benchmark_independent_recovery_amendment.json" \
  "${TARGET_ROOT}/reference_data/mohammadzadeh_2012/mv17b_a3_benchmark_independent_recovery_amendment.json"
chmod +x \
  "${PAYLOAD_TARGET}/tests/test_mohammadzadeh_mv17b_a3_benchmark_independent_recovery.py" \
  "${PAYLOAD_TARGET}/scripts/submit_mohammadzadeh_mv17b_a3_unity.sh" \
  "${PAYLOAD_TARGET}/scripts/unity_mohammadzadeh_mv17b_a3_prepare.sbatch" \
  "${PAYLOAD_TARGET}/scripts/unity_mohammadzadeh_mv17b_a3_recover_array.sbatch" \
  "${PAYLOAD_TARGET}/scripts/unity_mohammadzadeh_mv17b_a3_analyze.sbatch" \
  "${PAYLOAD_TARGET}/scripts/unity_mohammadzadeh_mv17b_a3_post.sbatch"

source "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV17B_FRESH_CYLINDER_JOB.env"
PYTHON_BIN="${MV17B_A3_PYTHON:-${MV17B_PYTHON:-}}"
test -x "${PYTHON_BIN}"
cd "${TARGET_ROOT}"
export PYTHONPATH="${TARGET_ROOT}:${PYTHONPATH:-}"
"${PYTHON_BIN}" -m py_compile \
  vgdsmc/mohammadzadeh_mv17b_a3_benchmark_independent_recovery.py \
  vgdsmc/mohammadzadeh_mv17b_fresh_cylinder_confirmation.py
"${PYTHON_BIN}" "${PAYLOAD_TARGET}/tests/test_mohammadzadeh_mv17b_a3_benchmark_independent_recovery.py"
MV17B_PROJECT_ROOT="${TARGET_ROOT}" MV17B_A3_PYTHON="${PYTHON_BIN}" \
  exec bash "${PAYLOAD_TARGET}/scripts/submit_mohammadzadeh_mv17b_a3_unity.sh"
