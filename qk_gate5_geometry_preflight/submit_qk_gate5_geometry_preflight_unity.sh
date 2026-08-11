#!/usr/bin/env bash
set -euo pipefail

BASE=/project/pi_roohie_umass_edu/Combustion/QK_GATE5_COUPLED
BASE_BUNDLE_REF=b831b4e4205aa0ad0660dcdc02a7ce0805642a8e
BASE_BUNDLE_SHA=520d455608f77e3408067de25482dd3a14ed7392322c51443d60c0bc469efffc
OVERLAY_REF=__OVERLAY_REF__
RAW_BASE="https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/$BASE_BUNDLE_REF/qk_gate5_shock_ignition_screen_bundle"
RAW_OVERLAY="https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/$OVERLAY_REF/qk_gate5_geometry_preflight"
TMP_GEOMETRY="$(mktemp -d)"
trap 'rm -rf "$TMP_GEOMETRY"' EXIT

mkdir -p "$BASE/releases" "$BASE/runs"

# Reconstruct the exact, previously validated Gate-5 base payload.
mkdir -p "$TMP_GEOMETRY/base_chunks"
for index in 00 01 02 03 04 05; do
  curl -fsSL "$RAW_BASE/chunks/chunk_$index?v=$BASE_BUNDLE_SHA" \
    -o "$TMP_GEOMETRY/base_chunks/chunk_$index"
done
BASE_ARCHIVE="$TMP_GEOMETRY/base_payload.tar.gz"
: > "$BASE_ARCHIVE"
for index in 00 01 02 03 04 05; do
  base64 --decode "$TMP_GEOMETRY/base_chunks/chunk_$index" >> "$BASE_ARCHIVE"
done
ACTUAL_BASE_SHA="$(sha256sum "$BASE_ARCHIVE" | awk '{print $1}')"
if [[ "$ACTUAL_BASE_SHA" != "$BASE_BUNDLE_SHA" ]]; then
  echo "Base checksum mismatch: expected=$BASE_BUNDLE_SHA actual=$ACTUAL_BASE_SHA" >&2
  exit 2
fi

BASE_RELEASE="$BASE/releases/$BASE_BUNDLE_SHA"
if [[ ! -d "$BASE_RELEASE" ]]; then
  BASE_STAGE="$BASE/releases/.stage_${BASE_BUNDLE_SHA}_$$"
  mkdir -p "$BASE_STAGE"
  tar -xzf "$BASE_ARCHIVE" -C "$BASE_STAGE"
  mv "$BASE_STAGE/payload" "$BASE_RELEASE"
  rmdir "$BASE_STAGE"
fi

# Fetch the immutable geometry overlay and verify every runtime file.
OVERLAY="$TMP_GEOMETRY/overlay"
mkdir -p "$OVERLAY"
curl -fsSL "$RAW_OVERLAY/MANIFEST.sha256" -o "$OVERLAY/MANIFEST.sha256"
while read -r expected relative; do
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || { echo "Invalid overlay hash" >&2; exit 3; }
  [[ "$relative" != /* && "$relative" != *..* ]] || { echo "Unsafe overlay path" >&2; exit 3; }
  mkdir -p "$OVERLAY/$(dirname "$relative")"
  curl -fsSL "$RAW_OVERLAY/$relative" -o "$OVERLAY/$relative"
done < "$OVERLAY/MANIFEST.sha256"
(
  cd "$OVERLAY"
  sha256sum -c MANIFEST.sha256
)
MANIFEST_SHA="$(sha256sum "$OVERLAY/MANIFEST.sha256" | awk '{print $1}')"
RELEASE_ID="gate5_geometry_${BASE_BUNDLE_SHA}_${MANIFEST_SHA:0:16}"
RELEASE="$BASE/releases/$RELEASE_ID"

if [[ ! -d "$RELEASE" ]]; then
  STAGE="$BASE/releases/.stage_${RELEASE_ID}_$$"
  mkdir -p "$STAGE"
  cp -a "$BASE_RELEASE/." "$STAGE/"
  while read -r _ relative; do
    mkdir -p "$STAGE/$(dirname "$relative")"
    cp "$OVERLAY/$relative" "$STAGE/$relative"
  done < "$OVERLAY/MANIFEST.sha256"
  python3 "$STAGE/tools/apply_gate5_geometry_patch.py" \
    --source "$STAGE/src/Viscous_Nozzle_GHS_commonfix.for"
  grep -q 'Gate 5 geometry preflight: boundary 6' \
    "$STAGE/src/Viscous_Nozzle_GHS_commonfix.for"
  chmod +x "$STAGE"/*.sh "$STAGE"/*.sbatch "$STAGE"/tools/*.py
  mv "$STAGE" "$RELEASE"
fi

grep -q 'Gate 5 geometry preflight: boundary 6' \
  "$RELEASE/src/Viscous_Nozzle_GHS_commonfix.for"
python3 "$RELEASE/tools/prepare_gate5_geometry_cases.py" \
  --template "$RELEASE/src/InputData.gHs.txt" --self-test
PYTHONPATH="$RELEASE/tools" python3 "$RELEASE/tools/validate_gate5_geometry.py" --self-test

SMOKE_ID="$(sbatch --parsable \
  --export=ALL,GATE5_GEOMETRY_RELEASE_ID="$RELEASE_ID" \
  "$RELEASE/run_gate5_geometry_smoke_unity.sbatch")"
SMOKE_ID="${SMOKE_ID%%;*}"

ARRAY_ID="$(sbatch --parsable --dependency="afterok:$SMOKE_ID" \
  --export=ALL,GATE5_GEOMETRY_RELEASE_ID="$RELEASE_ID" \
  "$RELEASE/run_gate5_geometry_array_unity.sbatch")"
ARRAY_ID="${ARRAY_ID%%;*}"

SUMMARY_ID="$(sbatch --parsable --dependency="afterok:$ARRAY_ID" \
  --export=ALL,GATE5_GEOMETRY_RELEASE_ID="$RELEASE_ID",GEOMETRY_ARRAY_JOB_ID="$ARRAY_ID" \
  "$RELEASE/run_gate5_geometry_summary_unity.sbatch")"
SUMMARY_ID="${SUMMARY_ID%%;*}"

RESULT_DIR="$BASE/runs/geometry_preflight_$ARRAY_ID"
cat > "$BASE/LAST_GATE5_GEOMETRY_PREFLIGHT.env" <<EOF
GEOMETRY_SMOKE_JOB_ID=$SMOKE_ID
GEOMETRY_ARRAY_JOB_ID=$ARRAY_ID
GEOMETRY_SUMMARY_JOB_ID=$SUMMARY_ID
OUTPUT_DIR=$RESULT_DIR
GEOMETRY_RELEASE_ID=$RELEASE_ID
BASE_PAYLOAD_SHA256=$BASE_BUNDLE_SHA
OVERLAY_MANIFEST_SHA256=$MANIFEST_SHA
EOF

echo "Submitted Gate-5 closed-duct chemistry-OFF geometry preflight."
echo "GEOMETRY_SMOKE_JOB_ID=$SMOKE_ID"
echo "GEOMETRY_ARRAY_JOB_ID=$ARRAY_ID"
echo "GEOMETRY_SUMMARY_JOB_ID=$SUMMARY_ID"
echo "OUTPUT_DIR=$RESULT_DIR"
echo "GEOMETRY_RELEASE_ID=$RELEASE_ID"
echo "Matrix: M1 target=3.5; T0=2500 K; p0=5 MPa; pb/p0={0.16,0.20,0.24}; chemistry=OFF."
echo "Guardrail: coarse shock-siting only; reacting DSMC is not launched."
squeue -j "$SMOKE_ID,$ARRAY_ID,$SUMMARY_ID" || true
