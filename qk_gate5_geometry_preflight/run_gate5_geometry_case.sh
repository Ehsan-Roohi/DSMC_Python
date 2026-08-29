#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUP="${1:?group result directory required}"
INDEX="${2:?case index required}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORK="$GROUP/tasks/task_$INDEX/work"
CASES="$GROUP/cases"
mkdir -p "$WORK" "$CASES"

CASE_NAME="$(PYTHONPATH="$ROOT/tools" "$PYTHON_BIN" -c \
  'import sys; from prepare_gate5_geometry_cases import CASES; print(CASES[int(sys.argv[1])][0])' \
  "$INDEX")"
CASE="$CASES/$CASE_NAME"

cp "$ROOT/src/Viscous_Nozzle_GHS_commonfix.for" "$WORK/"
cp "$ROOT/src/qk_chemistry.f90" "$ROOT/src/qk_nozzle_adapter.f90" "$WORK/"
cp "$ROOT/src/common.txt" "$ROOT/src/property.txt" "$WORK/"
"$PYTHON_BIN" "$ROOT/tools/prepare_gate5_geometry_cases.py" \
  --template "$ROOT/src/InputData.gHs.txt" --outroot "$CASES" --index "$INDEX"

(
  cd "$WORK"
  gfortran -c -O2 -g -ffree-line-length-none -fcheck=all -fbacktrace \
    -ffpe-trap=invalid,zero,overflow qk_chemistry.f90 qk_nozzle_adapter.f90
  gfortran -c -std=legacy -ffixed-line-length-none -O2 -g -fcheck=all \
    -fbacktrace -ffpe-trap=invalid,zero,overflow Viscous_Nozzle_GHS_commonfix.for
  gfortran qk_chemistry.o qk_nozzle_adapter.o \
    Viscous_Nozzle_GHS_commonfix.o -o nozzle_gate5_geometry.exe
)

cp "$WORK/nozzle_gate5_geometry.exe" "$WORK/common.txt" "$WORK/property.txt" "$CASE/"
cp "$WORK/common.txt" "$CASE/COMMON.TXT"
cp "$WORK/property.txt" "$CASE/Property.txt"
cp "$WORK/property.txt" "$CASE/PROPERTY.TXT"

(
  cd "$CASE"
  export GFORTRAN_UNBUFFERED_ALL=y
  printf '1\n' | timeout --signal=TERM --kill-after=60s 930m \
    ./nozzle_gate5_geometry.exe >run.log 2>&1
)

for required in QK_GATE5_EVENTS.txt QK_PRODUCTION_FLOW_FIELD.dat \
  QK_PRODUCTION_SPECIES_FIELD.dat QK_PRODUCTION_REACTION_FIELD.dat \
  QK_PRODUCTION_MONITOR.dat; do
  test -s "$CASE/$required"
done
echo "QK_GATE5_GEOMETRY_CASE_PASS index=$INDEX case=$CASE_NAME"
