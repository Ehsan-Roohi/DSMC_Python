#!/usr/bin/env bash
# Submit the immutable Gate-5 Mach/residence design stage to Unity.
# Run with `bash`; never source this strict-shell script.
set -euo pipefail

BASE=${BASE:-/project/pi_roohie_umass_edu/Combustion/QK_GATE5_COUPLED}
RESULT_DIR=${RESULT_DIR:-$BASE/runs/screen_62734194}
SOURCE_COMMIT=f375b11b8752cf0482d09498bdb414f8410807f0
ANALYZER_SHA256=fdeff12a01c8797f5aa7a24126414ed92472ed3bfe410f18422e8b51c0b3f3ff
DESIGNER_SHA256=543ab50776317dc3febb5d1bb9393b8bbeda28599e0cfe7367a1875c98662221
SBATCH_SHA256=b925414a0bcc7562399d68cb72c56bb59169109a8bc4c1c848670f9e941db1b1
RAW_BASE="https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/$SOURCE_COMMIT/qk_gate5_physics_prescreen"
RELEASE="$BASE/releases/gate5_mach_residence_$SOURCE_COMMIT"
ANALYZER="$RELEASE/gate5_physics_prescreen.py"
DESIGNER="$RELEASE/gate5_mach_residence_design.py"
SBATCH_FILE="$RELEASE/run_qk_gate5_mach_residence_design_unity.sbatch"

if [[ ! -d "$RESULT_DIR" ]]; then
  echo "ERROR: completed Gate-5 result directory not found: $RESULT_DIR" >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch is unavailable; run on a Unity login node." >&2
  exit 2
fi

mkdir -p "$RELEASE"
TMP_ANALYZER="$ANALYZER.tmp.$$"
TMP_DESIGNER="$DESIGNER.tmp.$$"
TMP_SBATCH="$SBATCH_FILE.tmp.$$"
cleanup() {
  rm -f "$TMP_ANALYZER" "$TMP_DESIGNER" "$TMP_SBATCH"
}
trap cleanup EXIT

curl -fsSL "$RAW_BASE/gate5_physics_prescreen.py" -o "$TMP_ANALYZER"
curl -fsSL "$RAW_BASE/gate5_mach_residence_design.py" -o "$TMP_DESIGNER"
curl -fsSL "$RAW_BASE/run_qk_gate5_mach_residence_design_unity.sbatch" -o "$TMP_SBATCH"
printf '%s  %s\n' "$ANALYZER_SHA256" "$TMP_ANALYZER" | sha256sum -c -
printf '%s  %s\n' "$DESIGNER_SHA256" "$TMP_DESIGNER" | sha256sum -c -
printf '%s  %s\n' "$SBATCH_SHA256" "$TMP_SBATCH" | sha256sum -c -
mv "$TMP_ANALYZER" "$ANALYZER"
mv "$TMP_DESIGNER" "$DESIGNER"
mv "$TMP_SBATCH" "$SBATCH_FILE"
chmod 0555 "$ANALYZER" "$DESIGNER" "$SBATCH_FILE"

SUBMITTED=$(sbatch --parsable \
  --export="ALL,GATE5_BASE=$BASE,GATE5_RESULT_DIR=$RESULT_DIR,GATE5_ANALYZER=$ANALYZER,GATE5_ANALYZER_SHA256=$ANALYZER_SHA256,GATE5_DESIGNER=$DESIGNER,GATE5_DESIGNER_SHA256=$DESIGNER_SHA256" \
  "$SBATCH_FILE")
JOB_ID=${SUBMITTED%%;*}
if [[ ! "$JOB_ID" =~ ^[0-9]+$ ]]; then
  echo "ERROR: unexpected sbatch response: $SUBMITTED" >&2
  exit 4
fi

OUTPUT_DIR="$BASE/runs/mach_residence_design_$JOB_ID"
ENV_FILE="$BASE/LAST_GATE5_MACH_RESIDENCE_DESIGN.env"
printf '%s\n' \
  "DESIGN_JOB_ID=$JOB_ID" \
  "SOURCE_COMMIT=$SOURCE_COMMIT" \
  "ANALYZER_SHA256=$ANALYZER_SHA256" \
  "DESIGNER_SHA256=$DESIGNER_SHA256" \
  "SOURCE_RESULT_DIR=$RESULT_DIR" \
  "OUTPUT_DIR=$OUTPUT_DIR" \
  > "$ENV_FILE"

echo "Submitted Gate-5 Mach/residence design: $JOB_ID"
echo "Expected output: $OUTPUT_DIR"
echo "State file: $ENV_FILE"
echo
echo "Status:"
echo "  sacct -j $JOB_ID --format=JobID,JobName%22,State,ExitCode,Elapsed,MaxRSS,NodeList"
echo "Log:"
echo "  tail -f $BASE/runs/mach-residence-$JOB_ID.out"
