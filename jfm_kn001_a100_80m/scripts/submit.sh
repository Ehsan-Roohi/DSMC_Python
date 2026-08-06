#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$ROOT/run_output}"
CASE_TABLE="$ROOT/cases/kn001_a100_80m.csv"
mkdir -p "$OUTPUT_ROOT" "$ROOT/slurm"
bash scripts/preflight_local.sh

COMMON_EXPORT="ALL,JFM_ROOT=$ROOT,JFM_OUTPUT_ROOT=$OUTPUT_ROOT,JFM_CASE_TABLE=$CASE_TABLE"
PREFLIGHT_JOB="$(sbatch --parsable \
  --job-name=jfm-k001-a100-mem \
  --partition=gpu --gpus=1 --constraint=a100-80g \
  --cpus-per-task=8 --mem=96G --time=02:00:00 \
  --output="$ROOT/slurm/jfm-k001-a100-preflight-%j.out" \
  --error="$ROOT/slurm/jfm-k001-a100-preflight-%j.err" \
  --export="$COMMON_EXPORT" scripts/gpu_preflight.slurm)"

RUN_JOB="$(sbatch --parsable \
  --dependency="afterok:$PREFLIGHT_JOB" --kill-on-invalid-dep=yes \
  --job-name=jfm-k001-a100 \
  --partition=gpu --gpus=1 --constraint=a100-80g \
  --cpus-per-task=8 --mem=96G --time=336:00:00 \
  --array='0-11%4' \
  --output="$ROOT/slurm/jfm-k001-a100-%A_%a.out" \
  --error="$ROOT/slurm/jfm-k001-a100-%A_%a.err" \
  --export="$COMMON_EXPORT" scripts/run_a100.slurm)"

SUMMARY_JOB="$(sbatch --parsable \
  --dependency="afterok:$RUN_JOB" --kill-on-invalid-dep=yes \
  --export="$COMMON_EXPORT" scripts/summarize.slurm)"

printf 'PREFLIGHT_JOB=%q\nRUN_JOB=%q\nSUMMARY_JOB=%q\nOUTPUT_ROOT=%q\n' \
  "$PREFLIGHT_JOB" "$RUN_JOB" "$SUMMARY_JOB" "$OUTPUT_ROOT" \
  > LAST_A100_SUBMISSION.env

echo "PREFLIGHT_JOB=$PREFLIGHT_JOB"
echo "RUN_JOB=$RUN_JOB"
echo "SUMMARY_JOB=$SUMMARY_JOB"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "Monitor: squeue -j ${PREFLIGHT_JOB},${RUN_JOB},${SUMMARY_JOB}"
