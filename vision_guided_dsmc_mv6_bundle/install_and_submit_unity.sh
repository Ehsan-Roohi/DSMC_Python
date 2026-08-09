#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="${1:-${MV6_TARGET_ROOT:-$PWD}}"

test -f "${TARGET_ROOT}/vgdsmc/mohammadzadeh_vision_mv3.py"
test -f "${TARGET_ROOT}/vgdsmc/mohammadzadeh_vision_mv4.py"
test -f "${TARGET_ROOT}/vgdsmc/mohammadzadeh_vision_mv5.py"
test -f "${TARGET_ROOT}/vgdsmc/mohammadzadeh_mv5_reference.py"

for relative in \
  vgdsmc/mohammadzadeh_architecture_screen.py \
  tests/test_mohammadzadeh_architecture_screen.py \
  reference_data/mohammadzadeh_2012/mv6_four_architecture_screen_protocol.json \
  scripts/submit_mohammadzadeh_architecture_screen_unity.sh \
  scripts/unity_mohammadzadeh_architecture_screen_task.sbatch \
  scripts/unity_mohammadzadeh_architecture_screen_post.sbatch; do
  install -D -m 0644 "${BUNDLE_ROOT}/payload/${relative}" "${TARGET_ROOT}/${relative}"
done
chmod +x \
  "${TARGET_ROOT}/scripts/submit_mohammadzadeh_architecture_screen_unity.sh" \
  "${TARGET_ROOT}/scripts/unity_mohammadzadeh_architecture_screen_task.sbatch" \
  "${TARGET_ROOT}/scripts/unity_mohammadzadeh_architecture_screen_post.sbatch"

cd "${TARGET_ROOT}"
python3 -m compileall -q vgdsmc/mohammadzadeh_architecture_screen.py
exec bash scripts/submit_mohammadzadeh_architecture_screen_unity.sh
