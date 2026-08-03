#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/pi_roohie_umass_edu/Combustion/QK_GATE2_UNITY
ENV_FILE="$ROOT/LAST_GATE2_JOB.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "No submitted Gate 2 job found at $ENV_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

echo "JOB_ID=$JOB_ID"
squeue -j "$JOB_ID" || true

if [[ -f "$OUT" ]]; then
  echo "--- stdout tail ---"
  tail -n 60 "$OUT"
fi
if [[ -f "$ERR" && -s "$ERR" ]]; then
  echo "--- stderr tail ---"
  tail -n 60 "$ERR"
fi

REPORT="$RESULT_DIR/validation_output/QK_GATE2_VALIDATION_REPORT.txt"
if [[ -f "$REPORT" ]]; then
  echo "--- final report ---"
  cat "$REPORT"
  if grep -q 'OVERALL=PASS' "$REPORT"; then
    echo "UNITY_QK_GATE2_PASS"
  else
    echo "UNITY_QK_GATE2_REPORT_PRESENT_BUT_NOT_PASS"
  fi
else
  echo "Final report not present yet: $REPORT"
fi
