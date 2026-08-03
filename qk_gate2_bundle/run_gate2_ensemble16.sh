#!/usr/bin/env bash
set -euo pipefail

BASE="${1:-$(pwd)/qk_gate2}"
PY="${PYTHON:-python3}"
FC="${FC:-gfortran}"
NPROC="${SLURM_CPUS_PER_TASK:-4}"
SEEDS=($(seq 56001 56016))
OUT="$BASE/validation_output_ensemble16"
FRUN="$BASE/fortran_runs_ensemble16"

cd "$BASE"
rm -rf "$OUT" "$FRUN"
mkdir -p "$OUT" "$FRUN"

"$FC" -O2 -std=f2008 -ffree-line-length-none \
  qk_chemistry.f90 qk_gate2_event_selftest.f90 \
  -o qk_gate2_event_selftest.exe \
  2> "$OUT/event_compile_warnings.txt"

"$FC" -O2 -std=f2008 -ffree-line-length-none \
  qk_chemistry.f90 qk_gate2_box.f90 \
  -o qk_gate2_box.exe \
  2> "$OUT/box_compile_warnings.txt"

./qk_gate2_event_selftest.exe | tee "$OUT/event_selftest_stdout.txt"
grep -q 'QK_GATE2_EVENT_SELFTEST_PASS' "$OUT/event_selftest_stdout.txt"

export BASE FRUN
printf '%s\n' "${SEEDS[@]}" | xargs -P "$NPROC" -I{} bash -lc '
  run="$FRUN/{}"
  mkdir -p "$run"
  cd "$run"
  "$BASE/qk_gate2_box.exe" "{}" > stdout.txt 2> stderr.txt
  grep -q "QK_GATE2_NUMBER_CHANGE_GATE_PASS" stdout.txt
'

"$PY" qk_python_gate2_reference.py \
  --seeds "${SEEDS[@]}" \
  --out "$OUT/python_gate2_reference_ensemble16.json" \
  > "$OUT/python_reference_stdout.txt"

grep -q 'PYTHON_QK_GATE2_REFERENCE_COMPLETE' "$OUT/python_reference_stdout.txt"

summaries=()
for seed in "${SEEDS[@]}"; do
  summaries+=("$FRUN/$seed/qk_gate2_summary.txt")
done

"$PY" compare_qk_gate2.py \
  --fortran "${summaries[@]}" \
  --python-reference "$OUT/python_gate2_reference_ensemble16.json" \
  --json-output "$OUT/qk_gate2_comparison_ensemble16.json" \
  --text-output "$OUT/QK_GATE2_ENSEMBLE16_REPORT.txt" \
  | tee "$OUT/comparison_stdout.txt"

grep -q 'QK_GATE2_CROSS_LANGUAGE_GATE_PASS' "$OUT/comparison_stdout.txt"
grep -q 'OVERALL=PASS' "$OUT/QK_GATE2_ENSEMBLE16_REPORT.txt"

echo 'QK_GATE2_ENSEMBLE16_PASS'
