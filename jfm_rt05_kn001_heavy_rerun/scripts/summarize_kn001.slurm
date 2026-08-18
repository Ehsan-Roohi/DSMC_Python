#!/bin/bash
#SBATCH --job-name=jfm-k001-sum
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-kn001-summary-%j.out
#SBATCH --error=slurm-kn001-summary-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?Submit from the package root}"
cd "$ROOT"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$ROOT/run_output}"
SUMMARY_DIR="$OUTPUT_ROOT/summary_kn001_heavy"
mkdir -p "$ROOT/.matplotlib" "$SUMMARY_DIR"
export MPLCONFIGDIR="$ROOT/.matplotlib"

mamba run -n dsmc-gpu python tools/summarize_ensembles.py \
  --input "$OUTPUT_ROOT/kn001_heavy" \
  --output "$SUMMARY_DIR" \
  --case-table cases/kn001_heavy.csv

test -s "$SUMMARY_DIR/ALL_ENSEMBLES_summary.csv"
test -s "$SUMMARY_DIR/ALL_ENSEMBLES_manifest.json"
echo "[OK] corrected Kn=0.01 three-seed summaries: $SUMMARY_DIR"
