#!/bin/bash
set -euo pipefail

REPOSITORY="Ehsan-Roohi/DSMC_Python"
REF="${JFM_GITHUB_REF:-f97b3d6a8e97e2bc41ad1444f8d2c32e9e5b152f}"
PROJECT_ROOT="${JFM_PROJECT_ROOT:-/project/pi_roohie_umass_edu/JFM_revision_2026}"
TARGET="${JFM_RESTARTABLE_FAST_ROOT:-$PROJECT_ROOT/JFM_SINGLE_SEED_RESTARTABLE_FAST}"
ARCHIVE_URL="https://codeload.github.com/${REPOSITORY}/tar.gz/${REF}"

for name in curl tar sbatch python sha256sum; do
  command -v "$name" >/dev/null || { echo "Missing required command: $name" >&2; exit 2; }
done

mkdir -p "$PROJECT_ROOT" "$TARGET"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

curl -fsSL "$ARCHIVE_URL" -o "$TMP_DIR/repository.tar.gz"
tar -xzf "$TMP_DIR/repository.tar.gz" -C "$TMP_DIR"
SOURCE_DIR="$(find "$TMP_DIR" -type d -path '*/jfm_single_seed_restartable_fast' -print -quit)"
[[ -n "$SOURCE_DIR" ]] || { echo "Restartable package not found at ref $REF" >&2; exit 4; }

cp -a "$SOURCE_DIR/." "$TARGET/"
cd "$TARGET"

python -m py_compile solver/*.py
HS_SHA="$(sha256sum solver/JFM_hs_dsmc_quarter.py | awk '{print $1}')"
RELAX_SHA="$(sha256sum solver/JFM_bgk_shakhov_quarter.py | awk '{print $1}')"
[[ "$HS_SHA" == "2c9e2f5119802b123f0335664a564085cfe77682ae4f4af4138dc31f5876b166" ]]
[[ "$RELAX_SHA" == "f2f97526942c53eca0af6dd9b94aca541f06951db4755a92cc8fb11e5e0e65a2" ]]

bash scripts/submit_checkpoint_fast.sh

echo "[OK] Restartable seven-case workflow submitted from pinned ref $REF"
echo "TARGET=$TARGET"
echo "No existing job was cancelled by this script."
