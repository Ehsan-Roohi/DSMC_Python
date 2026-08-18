#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$ROOT/run_output}"

bash scripts/preflight.sh

RUN_JOB="$(sbatch --parsable \
  --export=ALL,JFM_OUTPUT_ROOT="$OUTPUT_ROOT" \
  scripts/run_unity_kn001_heavy.slurm)"
SUMMARY_JOB="$(sbatch --parsable \
  --dependency=afterok:"$RUN_JOB" \
  --export=ALL,JFM_OUTPUT_ROOT="$OUTPUT_ROOT" \
  scripts/summarize_kn001.slurm)"

printf 'RUN_JOB=%q\nSUMMARY_JOB=%q\nOUTPUT_ROOT=%q\n' \
  "$RUN_JOB" "$SUMMARY_JOB" "$OUTPUT_ROOT" > LAST_KN001_SUBMISSION.env

echo "RUN_JOB=$RUN_JOB"
echo "SUMMARY_JOB=$SUMMARY_JOB"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "Monitor: squeue -j ${RUN_JOB},${SUMMARY_JOB}"
echo "Progress: squeue -r -j ${RUN_JOB}"
