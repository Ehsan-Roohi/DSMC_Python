#!/usr/bin/env bash
set -euo pipefail

REPO_RAW=https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/main/qk_gate4_nozzle_pilot_bundle
EXPECTED_SHA=37988e5f5a3334f082de198b37132d70ea275fd1aef869541d93cdd188447c15
BASE=/project/pi_roohie_umass_edu/Combustion/QK_GATE4_NOZZLE
TMPDIR_GATE4="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_GATE4"' EXIT

for index in 00 01 02 03 04; do
  curl -fsSL "$REPO_RAW/chunks/chunk_$index" -o "$TMPDIR_GATE4/chunk_$index"
done

ARCHIVE="$TMPDIR_GATE4/payload.tar.gz"
: > "$ARCHIVE"
for index in 00 01 02 03 04; do
  base64 --decode "$TMPDIR_GATE4/chunk_$index" >> "$ARCHIVE"
done

ACTUAL_SHA="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  echo "Checksum mismatch: expected=$EXPECTED_SHA actual=$ACTUAL_SHA" >&2
  exit 2
fi

mkdir -p "$BASE/releases" "$BASE/runs"
RELEASE="$BASE/releases/$EXPECTED_SHA"
if [[ ! -d "$RELEASE" ]]; then
  STAGE="$BASE/releases/.stage_$EXPECTED_SHA"
  mkdir -p "$STAGE"
  tar -xzf "$ARCHIVE" -C "$STAGE"
  mv "$STAGE/payload" "$RELEASE"
  rmdir "$STAGE"
fi
chmod +x "$RELEASE"/*.sh "$RELEASE"/*.sbatch "$RELEASE"/tools/*.py
ln -sfn "$RELEASE" "$BASE/payload"

JOB_ID="$(sbatch --parsable "$BASE/payload/run_gate4_nozzle_unity.sbatch")"
JOB_ID="${JOB_ID%%;*}"
OUT="$BASE/runs/slurm-$JOB_ID.out"
ERR="$BASE/runs/slurm-$JOB_ID.err"
RESULT_DIR="$BASE/runs/job_$JOB_ID"
cat > "$BASE/LAST_GATE4_NOZZLE_JOB.env" <<EOF
JOB_ID=$JOB_ID
OUT=$OUT
ERR=$ERR
RESULT_DIR=$RESULT_DIR
PAYLOAD_SHA256=$EXPECTED_SHA
EOF

echo "Submitted Gate 4B geometry repair + full-residence Q-K nozzle map."
echo "JOB_ID=$JOB_ID"
echo "Status: squeue -j $JOB_ID"
echo "Log: tail -f $OUT"
echo "Report: $RESULT_DIR/QK_GATE4_NOZZLE_PIPELINE_REPORT.txt"
