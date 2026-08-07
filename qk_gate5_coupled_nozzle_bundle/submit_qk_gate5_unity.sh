#!/usr/bin/env bash
set -euo pipefail

REPO_RAW=https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/main/qk_gate5_coupled_nozzle_bundle
EXPECTED_SHA=4701a22cd12296790229135ec00908422f1c43743c57672ba241b22847c442a9
BASE=/project/pi_roohie_umass_edu/Combustion/QK_GATE5_COUPLED
TMPDIR_GATE5="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_GATE5"' EXIT

for index in 00 01 02 03 04 05; do
  curl -fsSL "$REPO_RAW/chunks/chunk_$index?v=$EXPECTED_SHA" -o "$TMPDIR_GATE5/chunk_$index"
done
ARCHIVE="$TMPDIR_GATE5/payload.tar.gz"
: > "$ARCHIVE"
for index in 00 01 02 03 04 05; do
  base64 --decode "$TMPDIR_GATE5/chunk_$index" >> "$ARCHIVE"
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

# Linux include names are case-sensitive. The restored legacy solver requests
# COMMON.TXT and PROPERTY.TXT while the verified payload stores lowercase file
# names. Apply the compatibility edit after checksum verification so both a new
# release and an already cached release compile identically.
PATCHED_SOURCES=0
while IFS= read -r source_file; do
  if grep -Eq "(COMMON|PROPERTY)\.TXT" "$source_file"; then
    sed -i \
      -e "s/COMMON\.TXT/common.txt/g" \
      -e "s/PROPERTY\.TXT/property.txt/g" \
      "$source_file"
    PATCHED_SOURCES=$((PATCHED_SOURCES + 1))
  fi
done < <(find "$RELEASE" -type f \( -iname '*.for' -o -iname '*.f' -o -iname '*.f90' \) -print)

if grep -RIlE --include='*.for' --include='*.f' --include='*.f90' "(COMMON|PROPERTY)\.TXT" "$RELEASE" >/dev/null; then
  echo "Gate 5 preflight failed: unresolved uppercase legacy TXT include" >&2
  exit 3
fi
echo "GATE5_LEGACY_INCLUDE_CASEFIX_PASS patched_sources=$PATCHED_SOURCES"

chmod +x "$RELEASE"/*.sh "$RELEASE"/*.sbatch "$RELEASE"/tools/*.py
ln -sfn "$RELEASE" "$BASE/payload"

JOB_ID="$(sbatch --parsable "$BASE/payload/run_gate5_unity.sbatch")"
JOB_ID="${JOB_ID%%;*}"
OUT="$BASE/runs/slurm-$JOB_ID.out"
ERR="$BASE/runs/slurm-$JOB_ID.err"
RESULT_DIR="$BASE/runs/job_$JOB_ID"
cat > "$BASE/LAST_GATE5_JOB.env" <<EOF
JOB_ID=$JOB_ID
OUT=$OUT
ERR=$ERR
RESULT_DIR=$RESULT_DIR
PAYLOAD_SHA256=$EXPECTED_SHA
EOF

echo "Submitted Gate 5 coupled 2-D eight-species Q-K integration pilot."
echo "JOB_ID=$JOB_ID"
echo "Status: squeue -j $JOB_ID"
echo "Log: tail -f $OUT"
echo "Report: $RESULT_DIR/QK_GATE5_COUPLED_REPORT.json"
