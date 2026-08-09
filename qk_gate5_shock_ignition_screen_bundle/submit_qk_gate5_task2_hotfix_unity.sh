#!/usr/bin/env bash
set -euo pipefail

REPO_RAW=https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/b831b4e4205aa0ad0660dcdc02a7ce0805642a8e/qk_gate5_shock_ignition_screen_bundle
EXPECTED_SHA=520d455608f77e3408067de25482dd3a14ed7392322c51443d60c0bc469efffc
BASE=/project/pi_roohie_umass_edu/Combustion/QK_GATE5_COUPLED
TMPDIR_GATE5_FIX="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_GATE5_FIX"' EXIT

for index in 00 01 02 03 04 05; do
  curl -fsSL "$REPO_RAW/chunks/chunk_$index?v=$EXPECTED_SHA" \
    -o "$TMPDIR_GATE5_FIX/chunk_$index"
done

ARCHIVE="$TMPDIR_GATE5_FIX/payload.tar.gz"
: > "$ARCHIVE"
for index in 00 01 02 03 04 05; do
  base64 --decode "$TMPDIR_GATE5_FIX/chunk_$index" >> "$ARCHIVE"
done
ACTUAL_SHA="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  echo "Checksum mismatch: expected=$EXPECTED_SHA actual=$ACTUAL_SHA" >&2
  exit 2
fi

mkdir -p "$BASE/releases" "$BASE/runs"
RELEASE="$BASE/releases/$EXPECTED_SHA"
if [[ ! -d "$RELEASE" ]]; then
  STAGE="$BASE/releases/.stage_${EXPECTED_SHA}_$$"
  mkdir -p "$STAGE"
  tar -xzf "$ARCHIVE" -C "$STAGE"
  mv "$STAGE/payload" "$RELEASE"
  rmdir "$STAGE"
fi

grep -q 'INT(NCOL,KIND=8)' "$RELEASE/src/Viscous_Nozzle_GHS_commonfix.for"
if grep -q 'IDINT(NCOL)' "$RELEASE/src/Viscous_Nozzle_GHS_commonfix.for"; then
  echo "Unsafe 32-bit NCOL progress conversion remains" >&2
  exit 4
fi
chmod +x "$RELEASE"/*.sh "$RELEASE"/*.sbatch "$RELEASE"/tools/*.py

ENVFILE="$BASE/LAST_GATE5_SHOCK_IGNITION_SCREEN.env"
source "$ENVFILE"
ORIGINAL_ARRAY_JOB_ID="$ARRAY_JOB_ID"
ORIGINAL_RESULT_DIR="$RESULT_DIR"
if [[ "$ORIGINAL_RESULT_DIR" != "$BASE/runs/screen_$ORIGINAL_ARRAY_JOB_ID" ]]; then
  echo "Unexpected original result directory: $ORIGINAL_RESULT_DIR" >&2
  exit 5
fi

GOOD_CASES="$(find "$ORIGINAL_RESULT_DIR/cases" -mindepth 2 -maxdepth 2 \
  -name QK_GATE5_EVENTS.txt -size +0c | wc -l)"
if [[ "$GOOD_CASES" -ne 11 ]]; then
  echo "Expected 11 completed cases before task-2 repair; found $GOOD_CASES" >&2
  exit 6
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP="$ORIGINAL_RESULT_DIR/incomplete_task2_$STAMP"
mkdir -p "$BACKUP"
for target in \
  "$ORIGINAL_RESULT_DIR/tasks/task_2" \
  "$ORIGINAL_RESULT_DIR/cases/p5_r024_t1000_chem_on" \
  "$ORIGINAL_RESULT_DIR/task_2.log"; do
  if [[ -e "$target" ]]; then mv "$target" "$BACKUP/"; fi
done

FIX_ID="$(sbatch --parsable \
  --export=ALL,GATE5_PAYLOAD_SHA256="$EXPECTED_SHA",SCREEN_RESULT_DIR="$ORIGINAL_RESULT_DIR" \
  "$RELEASE/run_gate5_screen_task2_hotfix_unity.sbatch")"
FIX_ID="${FIX_ID%%;*}"

SUMMARY_ID="$(sbatch --parsable --dependency="afterok:$FIX_ID" \
  --export=ALL,GATE5_PAYLOAD_SHA256="$EXPECTED_SHA",SCREEN_ARRAY_JOB_ID="$ORIGINAL_ARRAY_JOB_ID" \
  "$RELEASE/run_gate5_screen_summary_unity.sbatch")"
SUMMARY_ID="${SUMMARY_ID%%;*}"

cat > "$BASE/LAST_GATE5_SHOCK_IGNITION_TASK2_HOTFIX.env" <<EOF
ORIGINAL_ARRAY_JOB_ID=$ORIGINAL_ARRAY_JOB_ID
ORIGINAL_RESULT_DIR=$ORIGINAL_RESULT_DIR
TASK2_HOTFIX_JOB_ID=$FIX_ID
SUMMARY_JOB_ID=$SUMMARY_ID
PAYLOAD_SHA256=$EXPECTED_SHA
BACKUP_DIR=$BACKUP
EOF

echo "Submitted task-2 clean rerun and dependent full summary."
echo "TASK2_HOTFIX_JOB_ID=$FIX_ID"
echo "SUMMARY_JOB_ID=$SUMMARY_ID"
echo "PAYLOAD_SHA256=$EXPECTED_SHA"
echo "RESULT_DIR=$ORIGINAL_RESULT_DIR"
echo "FAILED_OUTPUT_BACKUP=$BACKUP"
squeue -j "$FIX_ID,$SUMMARY_ID" || true
