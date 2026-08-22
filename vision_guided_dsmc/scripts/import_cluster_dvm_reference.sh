#!/usr/bin/env bash
set -euo pipefail

INPUT="${DVM_MOMENTS:-data/cavity_dvm/cavity_dvm_moments.dat}"
OUTPUT="${DVM_REFERENCE_OUT:-outputs/dvm/DVM65_reference.npz}"
CASE="${DVM_CASE:-DVM65}"
COLUMNS="${DVM_COLUMNS:-}"

args=(
  vgdsmc-import-dvm
  --input "$INPUT"
  --output "$OUTPUT"
  --case "$CASE"
  --knudsen "${DVM_KN:-0.075}"
  --lid-speed "${DVM_LID_SPEED:-0.1}"
  --wall-temperature "${DVM_WALL_T:-1.0}"
)

if [[ -n "$COLUMNS" ]]; then
  args+=(--columns "$COLUMNS")
fi

printf 'Importing %s -> %s\n' "$INPUT" "$OUTPUT"
"${args[@]}"
printf 'Created %s and %s\n' "$OUTPUT" "${OUTPUT%.npz}.json"
