#!/bin/bash
#SBATCH --job-name=jfm-k001-a100-sum
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail
ROOT="${JFM_ROOT:-${SLURM_SUBMIT_DIR:?}}"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$ROOT/run_output}"
cd "$ROOT"
mkdir -p "$OUTPUT_ROOT/summary"
mamba run -n dsmc-gpu python tools/summarize_six_seed.py \
  --input "$OUTPUT_ROOT/runs" \
  --output "$OUTPUT_ROOT/summary" \
  --case-table cases/kn001_a100_80m.csv
