#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="${1:-${MV5R_TARGET_ROOT:-$PWD}}"

test -f "${TARGET_ROOT}/vgdsmc/mohammadzadeh_mv5_reference.py"
test -f "${TARGET_ROOT}/vgdsmc/mohammadzadeh_vision_mv5.py"
test -f "${TARGET_ROOT}/vgdsmc/mohammadzadeh_architecture_screen.py"

for relative in \
  vgdsmc/mohammadzadeh_mv5_reference_stability_repair.py \
  reference_data/mohammadzadeh_2012/mv5_reference_stability_repair_protocol.json \
  scripts/unity_mohammadzadeh_mv5_reference_repair.sbatch \
  scripts/unity_mohammadzadeh_mv5_reference_repair_assemble.sbatch \
  scripts/submit_mohammadzadeh_mv5_repair_and_mv6_unity.sh \
  tests/test_mohammadzadeh_mv5_reference_stability_repair.py; do
  install -D -m 0644 "${BUNDLE_ROOT}/payload/${relative}" "${TARGET_ROOT}/${relative}"
done
chmod +x \
  "${TARGET_ROOT}/scripts/unity_mohammadzadeh_mv5_reference_repair.sbatch" \
  "${TARGET_ROOT}/scripts/unity_mohammadzadeh_mv5_reference_repair_assemble.sbatch" \
  "${TARGET_ROOT}/scripts/submit_mohammadzadeh_mv5_repair_and_mv6_unity.sh"

cd "${TARGET_ROOT}"
python3 -m compileall -q vgdsmc/mohammadzadeh_mv5_reference_stability_repair.py
exec bash scripts/submit_mohammadzadeh_mv5_repair_and_mv6_unity.sh
