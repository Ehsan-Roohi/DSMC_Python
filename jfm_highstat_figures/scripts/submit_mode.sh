#!/bin/bash
set -euo pipefail

MODE="${1:-}"
case "$MODE" in
  low|vram48|vram80) ;;
  *) echo "Usage: $0 {low|vram48|vram80}" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$ROOT/run_output}"
mkdir -p "$OUTPUT_ROOT/$MODE" "$ROOT/slurm"
bash scripts/preflight_local.sh

case "$MODE" in
  low)
    CONSTRAINT=2080ti
    ARRAY='0-29%20'
    RUN_TIME='336:00:00'
    RUN_MEM=48G
    CASE_TABLE=''
    ;;
  vram48)
    CONSTRAINT=vram48
    ARRAY='0-14%8'
    RUN_TIME='240:00:00'
    RUN_MEM=96G
    CASE_TABLE="$ROOT/cases/high48_40m.csv"
    ;;
  vram80)
    CONSTRAINT=vram80
    ARRAY='0-14%4'
    RUN_TIME='168:00:00'
    RUN_MEM=128G
    CASE_TABLE="$ROOT/cases/high80_40m.csv"
    ;;
esac

COMMON_EXPORT="ALL,JFM_ROOT=$ROOT,JFM_OUTPUT_ROOT=$OUTPUT_ROOT,JFM_ROUTE=$MODE"
if [[ -n "$CASE_TABLE" ]]; then
  COMMON_EXPORT="$COMMON_EXPORT,JFM_CASE_TABLE=$CASE_TABLE"
fi

PREFLIGHT_JOB="$(sbatch --parsable \
  --job-name="jfm-${MODE}-mem" \
  --partition=gpu --gpus=1 --constraint="$CONSTRAINT" \
  --cpus-per-task=8 --mem=96G --time=02:00:00 \
  --output="$ROOT/slurm/jfm-${MODE}-preflight-%j.out" \
  --error="$ROOT/slurm/jfm-${MODE}-preflight-%j.err" \
  --export="$COMMON_EXPORT" scripts/gpu_preflight.slurm)"

RUN_JOB="$(sbatch --parsable \
  --dependency="afterok:$PREFLIGHT_JOB" \
  --job-name="jfm-${MODE}-prod" \
  --partition=gpu --qos=long --gpus=1 --constraint="$CONSTRAINT" \
  --cpus-per-task=8 --mem="$RUN_MEM" --time="$RUN_TIME" \
  --array="$ARRAY" \
  --output="$ROOT/slurm/jfm-${MODE}-%A_%a.out" \
  --error="$ROOT/slurm/jfm-${MODE}-%A_%a.err" \
  --export="$COMMON_EXPORT" scripts/run_highstat.slurm)"

SUMMARY_JOB="$(sbatch --parsable \
  --dependency="afterok:$RUN_JOB" \
  --job-name="jfm-${MODE}-sum" \
  --partition=cpu --cpus-per-task=4 --mem=32G --time=08:00:00 \
  --output="$ROOT/slurm/jfm-${MODE}-summary-%j.out" \
  --error="$ROOT/slurm/jfm-${MODE}-summary-%j.err" \
  --export="$COMMON_EXPORT" scripts/summarize_highstat.slurm)"

ENV_FILE="$ROOT/LAST_SUBMISSION_${MODE}.env"
printf 'MODE=%q\nPREFLIGHT_JOB=%q\nRUN_JOB=%q\nSUMMARY_JOB=%q\nOUTPUT_ROOT=%q\n' \
  "$MODE" "$PREFLIGHT_JOB" "$RUN_JOB" "$SUMMARY_JOB" "$OUTPUT_ROOT" \
  > "$ENV_FILE"

echo "MODE=$MODE"
echo "PREFLIGHT_JOB=$PREFLIGHT_JOB"
echo "RUN_JOB=$RUN_JOB"
echo "SUMMARY_JOB=$SUMMARY_JOB"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "Monitor: squeue -j ${PREFLIGHT_JOB},${RUN_JOB},${SUMMARY_JOB}"
