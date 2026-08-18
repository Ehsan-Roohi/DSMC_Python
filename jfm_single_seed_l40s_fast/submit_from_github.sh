#!/bin/bash
set -euo pipefail

REPOSITORY="Ehsan-Roohi/DSMC_Python"
REF="${JFM_GITHUB_REF:-REPLACE_WITH_PINNED_COMMIT}"
PROJECT_ROOT="${JFM_PROJECT_ROOT:-/project/pi_roohie_umass_edu/JFM_revision_2026}"
TARGET="${JFM_SINGLE_SEED_ROOT:-$PROJECT_ROOT/JFM_SINGLE_SEED_L40S_FAST}"
ARCHIVE_URL="https://codeload.github.com/${REPOSITORY}/tar.gz/${REF}"

for name in curl tar sbatch; do
  command -v "$name" >/dev/null || { echo "Missing required command: $name" >&2; exit 2; }
done

mkdir -p "$PROJECT_ROOT" "$TARGET"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
curl -fsSL "$ARCHIVE_URL" -o "$TMP_DIR/repository.tar.gz"
tar -xzf "$TMP_DIR/repository.tar.gz" -C "$TMP_DIR"
SOURCE_DIR="$(find "$TMP_DIR" -type d -path '*/jfm_single_seed_l40s_fast' -print -quit)"
[[ -n "$SOURCE_DIR" ]] || { echo "Package not found at ref $REF" >&2; exit 4; }
cp -a "$SOURCE_DIR/." "$TARGET/"
cd "$TARGET"
bash scripts/submit_l40s.sh

echo
echo "Superseded-job cancellation is intentionally NOT automatic."
echo "After verifying the new job IDs above, run:"
echo "scancel 62597690_{1,2,4,5,7} 62597690_{8..20} 62597691 62643250 62643251 62643252"
