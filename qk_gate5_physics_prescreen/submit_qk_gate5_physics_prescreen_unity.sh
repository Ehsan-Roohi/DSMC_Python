#!/usr/bin/env bash
# Submit the immutable Gate-5 physics prescreen to Unity.
# Run this script with `bash`; do not source it into an interactive shell.
set -euo pipefail

BASE=${BASE:-/project/pi_roohie_umass_edu/Combustion/QK_GATE5_COUPLED}
RESULT_DIR=${RESULT_DIR:-$BASE/runs/screen_62734194}
SOURCE_COMMIT=29fa9774a0abd66d49679dd5b9f7b754d805e5ee
ANALYZER_SHA256=631454c67c0cfc35413b512867e5c1d7a4ba646f16c0ceb1fb2a26f5f7946bb0
SBATCH_SHA256=c579c09e40b1f146fbe6fce38c181bdb18accd1c508128d98575f5b74dddc48b
RAW_BASE="https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/$SOURCE_COMMIT/qk_gate5_physics_prescreen"
RELEASE="$BASE/releases/gate5_physics_$SOURCE_COMMIT"
ANALYZER="$RELEASE/gate5_physics_prescreen.py"
SBATCH_FILE="$RELEASE/run_qk_gate5_physics_prescreen_unity.sbatch"

if [[ ! -d "$RESULT_DIR" ]]; then
  echo "ERROR: completed Gate-5 result directory not found: $RESULT_DIR" >&2
  echo "Override it with: RESULT_DIR=/absolute/path bash <(curl ...)" >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch is unavailable; run this command on a Unity login node." >&2
  exit 2
fi

mkdir -p "$RELEASE"
TMP_ANALYZER="$ANALYZER.tmp.$$"
TMP_SBATCH="$SBATCH_FILE.tmp.$$"
cleanup() {
  rm -f "$TMP_ANALYZER" "$TMP_SBATCH"
}
trap cleanup EXIT

curl -fsSL "$RAW_BASE/gate5_physics_prescreen.py" -o "$TMP_ANALYZER"
curl -fsSL "$RAW_BASE/run_qk_gate5_physics_prescreen_unity.sbatch" -o "$TMP_SBATCH"
printf '%s  %s\n' "$ANALYZER_SHA256" "$TMP_ANALYZER" | sha256sum -c -
printf '%s  %s\n' "$SBATCH_SHA256" "$TMP_SBATCH" | sha256sum -c -
mv "$TMP_ANALYZER" "$ANALYZER"
mv "$TMP_SBATCH" "$SBATCH_FILE"
chmod 0555 "$ANALYZER" "$SBATCH_FILE"

SUBMITTED=$(sbatch --parsable \
  --export="ALL,GATE5_BASE=$BASE,GATE5_RESULT_DIR=$RESULT_DIR,GATE5_ANALYZER=$ANALYZER,GATE5_ANALYZER_SHA256=$ANALYZER_SHA256" \
  "$SBATCH_FILE")
JOB_ID=${SUBMITTED%%;*}
if [[ ! "$JOB_ID" =~ ^[0-9]+$ ]]; then
  echo "ERROR: unexpected sbatch response: $SUBMITTED" >&2
  exit 4
fi

OUTPUT_DIR="$BASE/runs/physics_prescreen_$JOB_ID"
ENV_FILE="$BASE/LAST_GATE5_PHYSICS_PRESCREEN.env"
printf '%s\n' \
  "PHYSICS_JOB_ID=$JOB_ID" \
  "SOURCE_COMMIT=$SOURCE_COMMIT" \
  "ANALYZER_SHA256=$ANALYZER_SHA256" \
  "SOURCE_RESULT_DIR=$RESULT_DIR" \
  "OUTPUT_DIR=$OUTPUT_DIR" \
  > "$ENV_FILE"

echo "Submitted Gate-5 physics prescreen: $JOB_ID"
echo "Source result: $RESULT_DIR"
echo "Expected output: $OUTPUT_DIR"
echo "State file: $ENV_FILE"
echo
echo "Status:"
echo "  sacct -j $JOB_ID --format=JobID,JobName%22,State,ExitCode,Elapsed,MaxRSS,NodeList"
echo "Log:"
echo "  tail -f $BASE/runs/physics-prescreen-$JOB_ID.out"
