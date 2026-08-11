#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTROOT="${1:?smoke result directory required}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORK="$OUTROOT/work"
CASES="$WORK/cases"
CASE="$CASES/m35_r020_chem_off_smoke"
mkdir -p "$WORK" "$CASES"

cp "$ROOT/src/Viscous_Nozzle_GHS_commonfix.for" "$WORK/"
cp "$ROOT/src/qk_chemistry.f90" "$ROOT/src/qk_nozzle_adapter.f90" "$WORK/"
cp "$ROOT/src/common.txt" "$ROOT/src/property.txt" "$WORK/"

"$PYTHON_BIN" "$ROOT/tools/apply_gate5_geometry_patch.py" --self-test
"$PYTHON_BIN" "$ROOT/tools/prepare_gate5_geometry_cases.py" \
  --template "$ROOT/src/InputData.gHs.txt" --self-test
PYTHONPATH="$ROOT/tools" "$PYTHON_BIN" \
  "$ROOT/tools/validate_gate5_geometry.py" --self-test
"$PYTHON_BIN" "$ROOT/tools/prepare_gate5_geometry_cases.py" \
  --template "$ROOT/src/InputData.gHs.txt" --outroot "$CASES" --smoke

(
  cd "$WORK"
  gfortran -c -O2 -g -ffree-line-length-none -fcheck=all -fbacktrace \
    -ffpe-trap=invalid,zero,overflow qk_chemistry.f90 qk_nozzle_adapter.f90
  gfortran -c -std=legacy -ffixed-line-length-none -O2 -g -fcheck=all \
    -fbacktrace -ffpe-trap=invalid,zero,overflow Viscous_Nozzle_GHS_commonfix.for
  gfortran -O0 -g -ffree-line-length-none -fcheck=all -fbacktrace \
    -ffpe-trap=invalid,zero,overflow "$ROOT/tools/test_qk_k0.f90" \
    qk_chemistry.o -o test_qk_k0.exe
  ./test_qk_k0.exe
  gfortran -O0 -g -ffree-line-length-none -fcheck=all -fbacktrace \
    -ffpe-trap=invalid,zero,overflow "$ROOT/tools/test_qk_thermochemistry.f90" \
    qk_chemistry.o -o test_qk_thermochemistry.exe
  ./test_qk_thermochemistry.exe
  gfortran -O0 -g -ffree-line-length-none -fcheck=all -fbacktrace \
    -ffpe-trap=invalid,zero,overflow "$ROOT/tools/test_qk_recomb_third_diss.f90" \
    qk_chemistry.o qk_nozzle_adapter.o -o test_qk_recomb_third_diss.exe
  ./test_qk_recomb_third_diss.exe
  gfortran qk_chemistry.o qk_nozzle_adapter.o \
    Viscous_Nozzle_GHS_commonfix.o -o nozzle_gate5_geometry.exe
)
echo "GATE5_GEOMETRY_SMOKE_COMPILE_PASS"

cp "$WORK/nozzle_gate5_geometry.exe" "$WORK/common.txt" "$WORK/property.txt" "$CASE/"
cp "$WORK/common.txt" "$CASE/COMMON.TXT"
cp "$WORK/property.txt" "$CASE/Property.txt"
cp "$WORK/property.txt" "$CASE/PROPERTY.TXT"

set +e
(
  cd "$CASE"
  export GFORTRAN_UNBUFFERED_ALL=y
  printf '1\n' | timeout --signal=TERM --kill-after=30s 50m \
    ./nozzle_gate5_geometry.exe >run.log 2>&1
)
RUN_STATUS=$?
set -e
echo "GATE5_GEOMETRY_SMOKE_RUN_STATUS=$RUN_STATUS"

PYTHONPATH="$ROOT/tools" "$PYTHON_BIN" "$ROOT/tools/validate_gate5_geometry.py" \
  --smoke-case "$CASE" --run-status "$RUN_STATUS" \
  --out "$OUTROOT/QK_GATE5_GEOMETRY_SMOKE.json"
cp "$OUTROOT/QK_GATE5_GEOMETRY_SMOKE.json" \
  "$OUTROOT/QK_GATE5_GEOMETRY_SMOKE.txt"
echo "GATE5_GEOMETRY_SMOKE_PIPELINE_PASS"
