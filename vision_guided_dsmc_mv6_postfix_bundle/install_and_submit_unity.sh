#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_BUNDLE_ROOT="$(cd "${BUNDLE_ROOT}/../vision_guided_dsmc_mv6_bundle" && pwd)"
TARGET_ROOT="${1:-${MV6_POSTFIX_TARGET_ROOT:-$PWD}}"

test -f "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV5_REPAIR_MV6_JOB.env"
test -f "${TARGET_ROOT}/scripts/unity_mohammadzadeh_architecture_screen_post.sbatch"
test -f "${SOURCE_BUNDLE_ROOT}/payload/vgdsmc/mohammadzadeh_architecture_screen.py"
test -f "${SOURCE_BUNDLE_ROOT}/payload/tests/test_mohammadzadeh_architecture_screen.py"

cd "${TARGET_ROOT}"
source LAST_MOHAMMADZADEH_MV5_REPAIR_MV6_JOB.env

test -x "${MV6_VENV_DIR}/bin/python"
test -d "${MV6_OUTPUT_ROOT}/tasks"

if [[ -e "${MV6_OUTPUT_ROOT}/summary.json" ]]; then
  echo "Refusing post-only rerun: MV6 summary already exists" >&2
  exit 2
fi

TASK_COUNT="$(find "${MV6_OUTPUT_ROOT}/tasks" -mindepth 3 -maxdepth 3 \
  -name summary.json -type f | wc -l)"
if [[ "${TASK_COUNT}" -ne 12 ]]; then
  echo "Expected 12 completed MV6 task summaries; found ${TASK_COUNT}" >&2
  exit 3
fi

install -D -m 0644 \
  "${SOURCE_BUNDLE_ROOT}/payload/vgdsmc/mohammadzadeh_architecture_screen.py" \
  "${TARGET_ROOT}/vgdsmc/mohammadzadeh_architecture_screen.py"
install -D -m 0644 \
  "${SOURCE_BUNDLE_ROOT}/payload/tests/test_mohammadzadeh_architecture_screen.py" \
  "${TARGET_ROOT}/tests/test_mohammadzadeh_architecture_screen.py"

source "${MV6_VENV_DIR}/bin/activate"
python -m compileall -q vgdsmc/mohammadzadeh_architecture_screen.py
if python -c 'import pytest' 2>/dev/null; then
  python -m pytest -q tests/test_mohammadzadeh_architecture_screen.py \
    -k metric_tree_comparison_tolerates_only_cpu_reduction_roundoff
fi

POST_EXPORTS="ALL,MV6_REPO_ROOT=${TARGET_ROOT},MV6_OUTPUT_ROOT=${MV6_OUTPUT_ROOT},MV6_VENV_DIR=${MV6_VENV_DIR}"
POST_JOB_ID="$(sbatch --parsable --export="${POST_EXPORTS}" \
  scripts/unity_mohammadzadeh_architecture_screen_post.sbatch)"

ENV_FILE="${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV6_POSTFIX_JOB.env"
printf 'MV6_POSTFIX_JOB_ID=%q\nMV6_PREVIOUS_POST_JOB_ID=%q\nMV6_OUTPUT_ROOT=%q\nMV6_VENV_DIR=%q\n' \
  "${POST_JOB_ID}" "${MV6_POST_JOB_ID}" "${MV6_OUTPUT_ROOT}" "${MV6_VENV_DIR}" \
  > "${ENV_FILE}"

echo "Installed hardened MV6 baseline verifier"
echo "Submitted postprocessor only: ${POST_JOB_ID}"
echo "Saved: ${ENV_FILE}"
