#!/usr/bin/env bash
set -euo pipefail

REPO_RAW=https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/main/qk_gate5_coupled_nozzle_bundle
EXPECTED_SHA=262013ce1e29a3cad89cb6ef5369918049079ba18904f60c446a2e17f3ad6057
BASE=/project/pi_roohie_umass_edu/Combustion/QK_GATE5_COUPLED
TMPDIR_GATE5="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_GATE5"' EXIT

for index in 00 01 02 03 04 05; do
  curl -fsSL "$REPO_RAW/chunks/chunk_$index?v=$EXPECTED_SHA" \
    -o "$TMPDIR_GATE5/chunk_$index"
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
  STAGE="$BASE/releases/.stage_${EXPECTED_SHA}_$$"
  mkdir -p "$STAGE"
  tar -xzf "$ARCHIVE" -C "$STAGE"
  mv "$STAGE/payload" "$RELEASE"
  rmdir "$STAGE"
fi

grep -q "DO 400 N=1,NACTIVE" \
  "$RELEASE/src/Viscous_Nozzle_GHS_commonfix.for"
grep -q "GATE5_ACTIVE_OUTPUT_CELLS" \
  "$RELEASE/src/Viscous_Nozzle_GHS_commonfix.for"
grep -q "third_body_eligible" "$RELEASE/src/qk_chemistry.f90"
grep -q "GATE5_K0_TWO_BODY_INDEX_SAFETY_PASS" \
  "$RELEASE/tools/test_qk_k0.f90"
grep -q "GATE5_RECOMB_THIRD_BODY_DISSOCIATION_PASS" \
  "$RELEASE/tools/test_qk_recomb_third_diss.f90"
grep -q "GATE5_THERMOCHEMISTRY_DATUM_PASS" \
  "$RELEASE/tools/test_qk_thermochemistry.f90"
grep -q "qk_reaction_heat(r)" "$RELEASE/src/qk_chemistry.f90"
grep -q "INVALID MOVE2 BOUNDARY" \
  "$RELEASE/src/Viscous_Nozzle_GHS_commonfix.for"
grep -q "parents=\[i,j,k\]" \
  "$RELEASE/src/qk_nozzle_adapter.f90"
if grep -Eq "INCLUDE '(COMMON|PROPERTY)\.TXT'" \
  "$RELEASE/src/Viscous_Nozzle_GHS_commonfix.for"; then
  echo "ERROR: unresolved case-sensitive legacy include" >&2
  exit 3
fi

chmod +x "$RELEASE"/*.sh "$RELEASE"/*.sbatch "$RELEASE"/tools/*.py
ln -sfn "$RELEASE" "$BASE/payload"

PASS_MARKER="$BASE/PREFLIGHT_PASS_${EXPECTED_SHA}.env"
rm -f "$PASS_MARKER"
JOB_ID="$(sbatch --parsable \
  --export=ALL,GATE5_PAYLOAD_SHA256="$EXPECTED_SHA" \
  "$BASE/payload/run_gate5_preflight_unity.sbatch")"
JOB_ID="${JOB_ID%%;*}"
OUT="$BASE/runs/preflight-slurm-$JOB_ID.out"
ERR="$BASE/runs/preflight-slurm-$JOB_ID.err"
RESULT_DIR="$BASE/runs/preflight_job_$JOB_ID"
cat > "$BASE/LAST_GATE5_PREFLIGHT_JOB.env" <<EOF
JOB_ID=$JOB_ID
OUT=$OUT
ERR=$ERR
RESULT_DIR=$RESULT_DIR
PAYLOAD_SHA256=$EXPECTED_SHA
PASS_MARKER=$PASS_MARKER
EOF

echo "Submitted the dense 5-bar Gate 5 preflight; the eight-case run remains locked."
echo "JOB_ID=$JOB_ID"
echo "Status: squeue -j $JOB_ID"
echo "Log: tail -f $OUT"
echo "Report: $RESULT_DIR/QK_GATE5_PREFLIGHT_REPORT.json"
