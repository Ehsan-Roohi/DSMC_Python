#!/usr/bin/env bash
set -euo pipefail

REPO_RAW=https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/main/qk_gate5_long_pair_bundle
EXPECTED_SHA=7d7833dc8ee9ab1dff8483da166811ad27d2ec7ab68e80e139eec958bea4e7e0
BASE=/project/pi_roohie_umass_edu/Combustion/QK_GATE5_COUPLED
TMPDIR_GATE5_LONG="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_GATE5_LONG"' EXIT

for index in 00 01 02 03 04 05; do
  curl -fsSL "$REPO_RAW/chunks/chunk_$index?v=$EXPECTED_SHA" \
    -o "$TMPDIR_GATE5_LONG/chunk_$index"
done

ARCHIVE="$TMPDIR_GATE5_LONG/payload.tar.gz"
: > "$ARCHIVE"
for index in 00 01 02 03 04 05; do
  base64 --decode "$TMPDIR_GATE5_LONG/chunk_$index" >> "$ARCHIVE"
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

grep -q '2400.*!NPTT' "$RELEASE/tools/prepare_gate5_cases.py"
grep -q '1200.*!NPS' "$RELEASE/tools/prepare_gate5_cases.py"
grep -q '"steps": 240000' "$RELEASE/tools/validate_gate5.py"
grep -q 'QK_GATE5_LONG_PRODUCTION_FLOW_SHOCK_COMBUSTION_PASS' "$RELEASE/tools/validate_gate5.py"
grep -q 'QK_PRODUCTION_FLOW_FIELD.dat' "$RELEASE/src/Viscous_Nozzle_GHS_commonfix.for"
grep -q 'QK_PRODUCTION_REACTION_FIELD.dat' "$RELEASE/src/Viscous_Nozzle_GHS_commonfix.for"
grep -q 'GATE5_PRESCRIBED_RESERVOIR_BOUNDARIES' "$RELEASE/src/Viscous_Nozzle_GHS_commonfix.for"
grep -q 'FATAL GATE5: PARTICLE RUNAWAY' "$RELEASE/src/Viscous_Nozzle_GHS_commonfix.for"
if sed -n '/SUBROUTINE ENTER2/,/SUBROUTINE REFLECT2/p' \
  "$RELEASE/src/Viscous_Nozzle_GHS_commonfix.for" | \
  grep -qE 'NSMP\.LT\.100|CALL PROPERTIES\(mc\)'; then
  echo 'Unsafe sampled-cell boundary feedback found in ENTER2' >&2
  exit 4
fi
grep -q '"30"' "$RELEASE/tools/prepare_gate5_preflight.py"
chmod +x "$RELEASE"/*.sh "$RELEASE"/*.sbatch "$RELEASE"/tools/*.py
ln -sfn "$RELEASE" "$BASE/payload"

PRE_ID="$(sbatch --parsable \
  --export=ALL,GATE5_PAYLOAD_SHA256="$EXPECTED_SHA" \
  "$RELEASE/run_gate5_preflight_unity.sbatch")"
PRE_ID="${PRE_ID%%;*}"

FULL_ID="$(sbatch --parsable --dependency="afterok:$PRE_ID" \
  --export=ALL,GATE5_PAYLOAD_SHA256="$EXPECTED_SHA" \
  "$RELEASE/run_gate5_unity.sbatch")"
FULL_ID="${FULL_ID%%;*}"

PRE_OUT="$BASE/runs/preflight-slurm-$PRE_ID.out"
PRE_ERR="$BASE/runs/preflight-slurm-$PRE_ID.err"
PRE_RESULT="$BASE/runs/preflight_job_$PRE_ID"
FULL_OUT="$BASE/runs/slurm-$FULL_ID.out"
FULL_ERR="$BASE/runs/slurm-$FULL_ID.err"
FULL_RESULT="$BASE/runs/job_$FULL_ID"

cat > "$BASE/LAST_GATE5_LONG_PAIR.env" <<EOF
PREFLIGHT_JOB_ID=$PRE_ID
PREFLIGHT_OUT=$PRE_OUT
PREFLIGHT_ERR=$PRE_ERR
PREFLIGHT_RESULT_DIR=$PRE_RESULT
JOB_ID=$FULL_ID
OUT=$FULL_OUT
ERR=$FULL_ERR
RESULT_DIR=$FULL_RESULT
PAYLOAD_SHA256=$EXPECTED_SHA
EOF

echo "Submitted Gate 5 LONG production pair (240,000 steps/case)."
echo "PREFLIGHT_JOB_ID=$PRE_ID"
echo "FULL_JOB_ID=$FULL_ID"
echo "PAYLOAD_SHA256=$EXPECTED_SHA"
echo "PREFLIGHT_LOG=$PRE_OUT"
echo "FULL_LOG=$FULL_OUT"
echo "RESULT_DIR=$FULL_RESULT"
echo "Expected full runtime: approximately 3.5-5 hours after preflight."
squeue -j "$PRE_ID,$FULL_ID" || true
