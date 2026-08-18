#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$ROOT/run_output}"

for command_name in sbatch mamba python; do
  command -v "$command_name" >/dev/null || {
    echo "Missing required command: $command_name" >&2
    exit 2
  }
done

test -w "$ROOT"
mkdir -p "$OUTPUT_ROOT/kn001_heavy" "$OUTPUT_ROOT/summary_kn001_heavy" "$ROOT/.matplotlib"
test -w "$OUTPUT_ROOT/kn001_heavy"

python -m py_compile \
  JFM_hs_dsmc_quarter_22m.py \
  JFM_bgk_shakhov_quarter_22m.py \
  tools/summarize_ensembles.py
bash -n scripts/run_unity_kn001_heavy.slurm
bash -n scripts/summarize_kn001.slurm
bash -n scripts/submit.sh

ROWS="$(awk 'END{print NR-1}' cases/kn001_heavy.csv)"
[[ "$ROWS" == "9" ]]
[[ "$(awk -F, 'NR>1 {print $2}' cases/kn001_heavy.csv | sort -u | tr '\n' ' ')" == "BGK HS SHAKHOV " ]]
[[ "$(awk -F, 'NR>1 {print $3}' cases/kn001_heavy.csv | sort -u)" == "0.01" ]]
[[ "$(awk -F, 'NR>1 {print $4}' cases/kn001_heavy.csv | sort -u)" == "0.5" ]]
[[ "$(awk -F, 'NR>1 {print $5}' cases/kn001_heavy.csv | sort -nu | tr '\n' ' ')" == "42 271828 314159 " ]]

mamba run -n dsmc-gpu python -c \
  'import cupy, numba, numpy, matplotlib; print("dsmc-gpu imports OK")'

echo "[OK] preflight passed: 9 heavy independent runs"
echo "Output root: $OUTPUT_ROOT"
