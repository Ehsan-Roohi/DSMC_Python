#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/pi_roohie_umass_edu/Combustion/QK_GATE3B_LIVE
source "$ROOT/LAST_GATE3B_JOB.env"
echo "JOB_ID=$JOB_ID"
squeue -j "$JOB_ID" || true
[[ -f "$OUT" ]] && { echo '--- stdout ---'; tail -n 120 "$OUT"; }
[[ -s "$ERR" ]] && { echo '--- stderr ---'; tail -n 120 "$ERR"; }
REPORT="$RESULT_DIR/validation_output/QK_GATE3B_VALIDATION_REPORT.txt"
if [[ -f "$REPORT" ]]; then
  echo '--- report ---'
  cat "$REPORT"
  grep -q 'OVERALL=PASS' "$REPORT" && echo 'UNITY_QK_GATE3B_PASS'
else
  echo "Report not present yet: $REPORT"
fi
