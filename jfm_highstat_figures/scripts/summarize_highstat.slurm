#!/bin/bash
set -euo pipefail

ROOT="${JFM_ROOT:-${SLURM_SUBMIT_DIR:?}}"
ROUTE="${JFM_ROUTE:?JFM_ROUTE is required}"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$ROOT/run_output}"
cd "$ROOT"

if [[ "$ROUTE" == "low" ]]; then
  # shellcheck disable=SC1090
  source "$OUTPUT_ROOT/$ROUTE/mode.env"
  if [[ "$JFM_LOW_MODE" == "40m" ]]; then
    CASE_FILE="$ROOT/cases/low_40m.csv"
  else
    CASE_FILE="$ROOT/cases/low_20m_split.csv"
  fi
else
  CASE_FILE="${JFM_CASE_TABLE:?JFM_CASE_TABLE is required}"
fi

SUMMARY_DIR="$OUTPUT_ROOT/$ROUTE/summary"
mkdir -p "$SUMMARY_DIR" "$ROOT/.matplotlib"
export MPLCONFIGDIR="$ROOT/.matplotlib"

mamba run -n dsmc-gpu python tools/summarize_highstat.py \
  --input "$OUTPUT_ROOT/$ROUTE/runs" \
  --output "$SUMMARY_DIR" \
  --case-table "$CASE_FILE" \
  --route "$ROUTE"

test -s "$SUMMARY_DIR/ALL_HIGHSTAT_ENSEMBLES.csv"
test -s "$SUMMARY_DIR/ALL_HIGHSTAT_ENSEMBLES.json"
echo "[OK] $ROUTE summary complete: $SUMMARY_DIR"
