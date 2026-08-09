#!/usr/bin/env bash
set -euo pipefail

REPO_RAW=https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/agent/qk-gate5-shock-ignition-screen/qk_gate5_shock_ignition_screen_bundle
EXPECTED_SHA=74358360021f13fed786293271ab8c7a3b09e387e5495cfb9b78f1046e056778
BASE=/project/pi_roohie_umass_edu/Combustion/QK_GATE5_COUPLED
TMPDIR_GATE5_SCREEN="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_GATE5_SCREEN"' EXIT

for index in 00 01 02 03 04 05; do
  curl -fsSL "$REPO_RAW/chunks/chunk_$index?v=$EXPECTED_SHA" \
    -o "$TMPDIR_GATE5_SCREEN/chunk_$index"
done

ARCHIVE="$TMPDIR_GATE5_SCREEN/payload.tar.gz"
: > "$ARCHIVE"
for index in 00 01 02 03 04 05; do
  base64 --decode "$TMPDIR_GATE5_SCREEN/chunk_$index" >> "$ARCHIVE"
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

grep -q 'TBACK_K = 300.0' "$RELEASE/tools/prepare_gate5_cases.py"
grep -q 'TEMPERATURES_K = (1000.0, 1250.0, 1500.0, 1750.0)' "$RELEASE/tools/prepare_gate5_cases.py"
grep -q 'BACK_PRESSURE_RATIOS = (0.12, 0.18, 0.24)' "$RELEASE/tools/prepare_gate5_cases.py"
grep -q '600.*!NPTT' "$RELEASE/tools/prepare_gate5_cases.py"
grep -q '300.*!NPS' "$RELEASE/tools/prepare_gate5_cases.py"
grep -q 'COMMON /GATE5BC/ FTMPB' "$RELEASE/src/common.txt"
grep -q 'READ(222,\*,IOSTAT=IOSQK) FTMPB' "$RELEASE/src/Viscous_Nozzle_GHS_commonfix.for"
grep -q 'TEMP=FTMPB' "$RELEASE/src/Viscous_Nozzle_GHS_commonfix.for"
grep -q 'fsp=\[1.0/3.0.*1.0/6.0.*1.0/2.0\]' "$RELEASE/src/qk_nozzle_adapter.f90"
grep -q '#SBATCH --array=0-11%4' "$RELEASE/run_gate5_screen_array_unity.sbatch"
grep -q 'candidate_for_off_on_pair' "$RELEASE/tools/validate_gate5.py"
grep -q 'max_particles < 1800000' "$RELEASE/tools/validate_gate5_preflight.py"
if sed -n '/SUBROUTINE ENTER2/,/SUBROUTINE REFLECT2/p' \
  "$RELEASE/src/Viscous_Nozzle_GHS_commonfix.for" | \
  grep -qE 'POUT/\(BOLTZ\*FTMP\)|NSMP\.LT\.100|CALL PROPERTIES\(mc\)'; then
  echo 'Unsafe or non-independent receiver boundary found in ENTER2' >&2
  exit 4
fi
chmod +x "$RELEASE"/*.sh "$RELEASE"/*.sbatch "$RELEASE"/tools/*.py
ln -sfn "$RELEASE" "$BASE/payload"

PRE_ID="$(sbatch --parsable \
  --export=ALL,GATE5_PAYLOAD_SHA256="$EXPECTED_SHA" \
  "$RELEASE/run_gate5_preflight_unity.sbatch")"
PRE_ID="${PRE_ID%%;*}"

ARRAY_ID="$(sbatch --parsable --dependency="afterok:$PRE_ID" \
  --export=ALL,GATE5_PAYLOAD_SHA256="$EXPECTED_SHA" \
  "$RELEASE/run_gate5_screen_array_unity.sbatch")"
ARRAY_ID="${ARRAY_ID%%;*}"

SUMMARY_ID="$(sbatch --parsable --dependency="afterok:$ARRAY_ID" \
  --export=ALL,GATE5_PAYLOAD_SHA256="$EXPECTED_SHA",SCREEN_ARRAY_JOB_ID="$ARRAY_ID" \
  "$RELEASE/run_gate5_screen_summary_unity.sbatch")"
SUMMARY_ID="${SUMMARY_ID%%;*}"

RESULT_DIR="$BASE/runs/screen_$ARRAY_ID"
cat > "$BASE/LAST_GATE5_SHOCK_IGNITION_SCREEN.env" <<EOF
PREFLIGHT_JOB_ID=$PRE_ID
ARRAY_JOB_ID=$ARRAY_ID
SUMMARY_JOB_ID=$SUMMARY_ID
RESULT_DIR=$RESULT_DIR
PAYLOAD_SHA256=$EXPECTED_SHA
EOF

echo "Submitted Gate 5 shock-triggered ignition screen."
echo "PREFLIGHT_JOB_ID=$PRE_ID"
echo "ARRAY_JOB_ID=$ARRAY_ID"
echo "SUMMARY_JOB_ID=$SUMMARY_ID"
echo "PAYLOAD_SHA256=$EXPECTED_SHA"
echo "RESULT_DIR=$RESULT_DIR"
echo "Matrix: T0={1000,1250,1500,1750} K; pb/p0={0.12,0.18,0.24}; Tback=300 K; 2H2+O2+3Ar."
squeue -j "$PRE_ID,$ARRAY_ID,$SUMMARY_ID" || true
