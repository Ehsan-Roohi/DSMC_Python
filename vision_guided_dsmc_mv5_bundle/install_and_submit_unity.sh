#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="${1:-${MV5_TARGET_ROOT:-$PWD}}"

test -f "${TARGET_ROOT}/vgdsmc/mohammadzadeh_vision_mv3.py"
test -f "${TARGET_ROOT}/vgdsmc/mohammadzadeh_vision_mv4.py"
test -f "${TARGET_ROOT}/scripts/submit_mohammadzadeh_vision_mv3_unity.sh"

for relative in \
  vgdsmc/mohammadzadeh_mv5_reference.py \
  vgdsmc/mohammadzadeh_vision_mv5.py \
  tests/test_mohammadzadeh_vision_mv5.py \
  reference_data/mohammadzadeh_2012/mv5_confirmatory_protocol.json \
  scripts/submit_mohammadzadeh_vision_mv5_unity.sh \
  scripts/unity_mohammadzadeh_mv5_reference.sbatch \
  scripts/unity_mohammadzadeh_vision_mv5_task.sbatch \
  scripts/unity_mohammadzadeh_vision_mv5_post.sbatch; do
  install -D -m 0644 "${BUNDLE_ROOT}/payload/${relative}" "${TARGET_ROOT}/${relative}"
done
chmod +x \
  "${TARGET_ROOT}/scripts/submit_mohammadzadeh_vision_mv5_unity.sh" \
  "${TARGET_ROOT}/scripts/unity_mohammadzadeh_mv5_reference.sbatch" \
  "${TARGET_ROOT}/scripts/unity_mohammadzadeh_vision_mv5_task.sbatch" \
  "${TARGET_ROOT}/scripts/unity_mohammadzadeh_vision_mv5_post.sbatch"

cd "${TARGET_ROOT}"
python3 -m compileall -q \
  vgdsmc/mohammadzadeh_mv5_reference.py \
  vgdsmc/mohammadzadeh_vision_mv5.py
exec bash scripts/submit_mohammadzadeh_vision_mv5_unity.sh
