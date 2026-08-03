#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTROOT="${1:-$ROOT/gate3b_validation_run}"
NPROC="${NPROC:-8}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
NSEEDS="${NSEEDS:-32}"
SEED_START="${SEED_START:-58001}"
MIN_SEEDS="${MIN_SEEDS:-32}"

if (( NSEEDS < MIN_SEEDS )); then
  echo "ERROR: Gate 3B requires at least $MIN_SEEDS seeds; requested $NSEEDS" >&2
  exit 2
fi
mapfile -t SEEDS < <(seq "$SEED_START" "$((SEED_START + NSEEDS - 1))")

rm -rf "$OUTROOT"
mkdir -p "$OUTROOT/fortran_runs" "$OUTROOT/python_runs" \
  "$OUTROOT/validation_output" "$OUTROOT/bin"
OUTROOT="$(cd "$OUTROOT" && pwd)"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

[[ -n "$PYTHON_BIN" ]] || { echo "ERROR: Python not found" >&2; exit 3; }
"$PYTHON_BIN" -c 'import numpy, numba' || {
  echo "ERROR: Gate 3B requires Python with numpy and numba" >&2
  exit 4
}

gfortran -std=f2008 -O2 -ffree-line-length-none \
  "$ROOT/qk_chemistry.f90" "$ROOT/qk_gate3_induction.f90" \
  -o "$OUTROOT/bin/qk_gate3_induction.exe"

printf '%s\n' "${SEEDS[@]}" | xargs -n1 -P "$NPROC" bash -c '
  seed="$1"
  directory="$OUTROOT/fortran_runs/$seed"
  mkdir -p "$directory"
  cd "$directory"
  "$OUTROOT/bin/qk_gate3_induction.exe" "$seed" > stdout.txt 2> stderr.txt
' _

for seed in "${SEEDS[@]}"; do
  grep -q 'QK_GATE3_SHOCK_INDUCTION_PASS' \
    "$OUTROOT/fortran_runs/$seed/stdout.txt"
done

export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/qk_gate3b_numba_${SLURM_JOB_ID:-$$}"
mkdir -p "$NUMBA_CACHE_DIR"

# Warm the independent Numba implementation once before the parallel launch.
seed="${SEEDS[0]}"
directory="$OUTROOT/python_runs/$seed"
mkdir -p "$directory"
(cd "$directory" && "$PYTHON_BIN" "$ROOT/qk_python_gate3_reference.py" \
  --seeds "$seed" --out result.json > stdout.txt 2> stderr.txt)
grep -q 'PYTHON_QK_GATE3_REFERENCE_COMPLETE' "$directory/stdout.txt"

printf '%s\n' "${SEEDS[@]:1}" | xargs -n1 -P "$NPROC" bash -c '
  seed="$1"
  directory="$OUTROOT/python_runs/$seed"
  mkdir -p "$directory"
  cd "$directory"
  "$PYTHON_BIN" "$ROOT/qk_python_gate3_reference.py" \
    --seeds "$seed" --out result.json > stdout.txt 2> stderr.txt
' _

for seed in "${SEEDS[@]}"; do
  grep -q 'PYTHON_QK_GATE3_REFERENCE_COMPLETE' \
    "$OUTROOT/python_runs/$seed/stdout.txt"
done

"$PYTHON_BIN" - "$OUTROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "python_runs"
runs = []
shock_state = None
for directory in sorted(root.iterdir(), key=lambda path: int(path.name)):
    data = json.loads((directory / "result.json").read_text())
    shock_state = data["shock_state"]
    runs.extend(data["runs"])
output = {
    "scope": "live independent Python post-shock Q-K induction oracle",
    "shock_state": shock_state,
    "runs": runs,
}
(Path(sys.argv[1]) / "python_gate3b_live_reference.json").write_text(
    json.dumps(output, indent=2) + "\n"
)
PY

"$PYTHON_BIN" "$ROOT/compare_qk_gate3b.py" \
  --fortran-root "$OUTROOT/fortran_runs" \
  --python-json "$OUTROOT/python_gate3b_live_reference.json" \
  --outdir "$OUTROOT/validation_output" \
  --min-seeds "$MIN_SEEDS" \
  | tee "$OUTROOT/validation_output/comparison_stdout.txt"

grep -q 'OVERALL=PASS' \
  "$OUTROOT/validation_output/QK_GATE3B_VALIDATION_REPORT.txt"
grep -q 'QK_GATE3B_LIVE_CROSS_LANGUAGE_PASS' \
  "$OUTROOT/validation_output/comparison_stdout.txt"

if [[ -f "$ROOT/GATE3B_PROVENANCE.txt" ]]; then
  cp "$ROOT/GATE3B_PROVENANCE.txt" "$OUTROOT/"
fi

echo 'GHS_NOZZLE_QK_GATE3B_PIPELINE_PASS'
echo "RESULT_DIR=$OUTROOT"
