#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="${1:-${MV4_TARGET_ROOT:-$PWD}}"

test -f "${TARGET_ROOT}/vgdsmc/mohammadzadeh_vision_mv3.py"
test -f "${TARGET_ROOT}/scripts/submit_mohammadzadeh_vision_mv3_unity.sh"

for relative in \
  vgdsmc/mohammadzadeh_vision_mv4.py \
  tests/test_mohammadzadeh_vision_mv4.py \
  reference_data/mohammadzadeh_2012/mv4_stability_repair_protocol.json \
  scripts/submit_mohammadzadeh_vision_mv4_unity.sh \
  scripts/unity_mohammadzadeh_vision_mv4_task.sbatch \
  scripts/unity_mohammadzadeh_vision_mv4_post.sbatch; do
  install -D -m 0644 "${BUNDLE_ROOT}/payload/${relative}" "${TARGET_ROOT}/${relative}"
done
chmod +x \
  "${TARGET_ROOT}/scripts/submit_mohammadzadeh_vision_mv4_unity.sh" \
  "${TARGET_ROOT}/scripts/unity_mohammadzadeh_vision_mv4_task.sbatch" \
  "${TARGET_ROOT}/scripts/unity_mohammadzadeh_vision_mv4_post.sbatch"

cd "${TARGET_ROOT}"
python3 -m compileall -q vgdsmc/mohammadzadeh_vision_mv4.py
exec bash scripts/submit_mohammadzadeh_vision_mv4_unity.sh
