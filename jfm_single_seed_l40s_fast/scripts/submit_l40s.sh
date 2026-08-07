#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$ROOT/run_output}"
CASE_TABLE="$ROOT/cases/remaining4.csv"
FIG5_DEPENDENCY_IDS="${JFM_FIG5_DEPENDENCY_IDS:-62606478:62611438:62611907}"
mkdir -p "$OUTPUT_ROOT" "$ROOT/slurm"
bash scripts/preflight_local.sh

COMMON_EXPORT="ALL,JFM_ROOT=$ROOT,JFM_OUTPUT_ROOT=$OUTPUT_ROOT,JFM_CASE_TABLE=$CASE_TABLE"

PREFLIGHT_JOB="$(sbatch --parsable \
  --job-name=jfm-1seed-l40-mem --partition=gpu --gpus=1 --constraint=l40s \
  --cpus-per-task=8 --mem=64G --time=02:00:00 \
  --output="$ROOT/slurm/jfm-1seed-l40-preflight-%j.out" \
  --error="$ROOT/slurm/jfm-1seed-l40-preflight-%j.err" \
  --export="$COMMON_EXPORT" scripts/gpu_preflight.slurm)"

RUN_JOB="$(sbatch --parsable --dependency="afterok:$PREFLIGHT_JOB" \
  --kill-on-invalid-dep=yes --job-name=jfm-1seed-l40 \
  --partition=gpu --gpus=1 --constraint=l40s \
  --cpus-per-task=8 --mem=64G --time=168:00:00 --array=0-3%4 \
  --output="$ROOT/slurm/jfm-1seed-l40-%A_%a.out" \
  --error="$ROOT/slurm/jfm-1seed-l40-%A_%a.err" \
  --export="$COMMON_EXPORT" scripts/run_l40s.slurm)"

SUMMARY_JOB="$(sbatch --parsable \
  --dependency="afterok:$RUN_JOB:$FIG5_DEPENDENCY_IDS" \
  --kill-on-invalid-dep=yes --job-name=jfm-1seed-sum \
  --partition=cpu --cpus-per-task=4 --mem=32G --time=08:00:00 \
  --output="$ROOT/slurm/jfm-1seed-summary-%j.out" \
  --error="$ROOT/slurm/jfm-1seed-summary-%j.err" \
  --export="$COMMON_EXPORT" scripts/summarize_single.slurm)"

printf 'PREFLIGHT_JOB=%q\nRUN_JOB=%q\nSUMMARY_JOB=%q\nOUTPUT_ROOT=%q\n' \
  "$PREFLIGHT_JOB" "$RUN_JOB" "$SUMMARY_JOB" "$OUTPUT_ROOT" \
  > "$ROOT/LAST_SUBMISSION.env"

echo "PREFLIGHT_JOB=$PREFLIGHT_JOB"
echo "RUN_JOB=$RUN_JOB"
echo "SUMMARY_JOB=$SUMMARY_JOB"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "Monitor: squeue -j $PREFLIGHT_JOB,$RUN_JOB,$SUMMARY_JOB"
