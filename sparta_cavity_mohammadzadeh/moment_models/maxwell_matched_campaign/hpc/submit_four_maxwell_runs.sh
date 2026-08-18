#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUNDLE_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
DELIVERY_ROOT=${MAXWELL_MODEL_CAMPAIGN_ROOT:-/project/pi_roohie_umass_edu/CavityColdToHotIdentify/JFM_Maxwell_Matched_Campaign_20260817}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULT_ROOT="$DELIVERY_ROOT/moment_results_$STAMP"
mkdir -p "$RESULT_ROOT"

python3 "$BUNDLE_ROOT/r13/tests/test_r13_maxwell_contract.py"
python3 "$BUNDLE_ROOT/r26/tests/test_maxwell_contract.py"

R13_SUB=$(sbatch --parsable --array=0-1%2 \
  --export="ALL,MAXWELL_MODEL_BUNDLE_ROOT=$BUNDLE_ROOT,MAXWELL_MODEL_RESULT_ROOT=$RESULT_ROOT" \
  "$SCRIPT_DIR/r13_maxwell_pair.slurm")
R13_JOB=${R13_SUB%%;*}

R26_LOG=$(mktemp /tmp/r26-maxwell-submit.XXXXXX)
R26_MAXWELL_SEED_KN005="$BUNDLE_ROOT/r26/seeds/kn005_N40/last_accepted_state.npz" \
R26_MAXWELL_SEED_KN020="$BUNDLE_ROOT/r26/seeds/kn020_N20/last_accepted_state.npz" \
OUT_ROOT="$RESULT_ROOT/r26" bash "$BUNDLE_ROOT/r26/hpc/submit_maxwell_pair.sh" | tee "$R26_LOG"
R26_K005=$(awk -F= '$1=="r26_maxwell_kn005_job"{print $2}' "$R26_LOG")
R26_K020=$(awk -F= '$1=="r26_maxwell_kn020_job"{print $2}' "$R26_LOG")
R26_RESULT=$(awk -F= '$1=="result_root"{print $2}' "$R26_LOG")
rm -f "$R26_LOG"
[[ -n "$R26_K005" && -n "$R26_K020" && -n "$R26_RESULT" ]] || { echo "R26 submission parsing failed" >&2; exit 3; }

COLLECT_SUB=$(sbatch --parsable \
  --dependency="afterany:${R13_JOB}:${R26_K005}:${R26_K020}" \
  --export="ALL,MAXWELL_MODEL_BUNDLE_ROOT=$BUNDLE_ROOT,MAXWELL_MODEL_RESULT_ROOT=$RESULT_ROOT,MAXWELL_R26_RESULT_ROOT=$R26_RESULT,MAXWELL_MODEL_DELIVERY_ROOT=$DELIVERY_ROOT,MAXWELL_R13_JOB_ID=$R13_JOB,MAXWELL_R26_K005_JOB_ID=$R26_K005,MAXWELL_R26_K020_JOB_ID=$R26_K020" \
  "$SCRIPT_DIR/collect_four_maxwell_runs.slurm")
COLLECT_JOB=${COLLECT_SUB%%;*}

printf 'Submitted four Maxwell moment-model runs.\n'
printf 'R13 array:       %s (task 0: KnGu=.05; task 1: KnGu=.20)\n' "$R13_JOB"
printf 'R26 KnGu=.05:    %s\nR26 KnGu=.20:    %s\nCollector:       %s\n' "$R26_K005" "$R26_K020" "$COLLECT_JOB"
printf 'Result root:     %s\n' "$RESULT_ROOT"
printf 'Monitor: squeue -j %s,%s,%s,%s\n' "$R13_JOB" "$R26_K005" "$R26_K020" "$COLLECT_JOB"
