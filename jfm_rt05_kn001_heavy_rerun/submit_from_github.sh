#!/bin/bash
set -euo pipefail

REPOSITORY="Ehsan-Roohi/DSMC_Python"
# Immutable package snapshot containing all validated solver and workflow files.
REF="${JFM_GITHUB_REF:-65beda519fb4c40b9d9367c3108265313974cc48}"
PROJECT_ROOT="${JFM_PROJECT_ROOT:-/project/pi_roohie_umass_edu/JFM_revision_2026}"
TARGET="${JFM_RERUN_ROOT:-$PROJECT_ROOT/JFM_RT05_KN001_HEAVY_RERUN}"
ARCHIVE_URL="https://codeload.github.com/${REPOSITORY}/tar.gz/${REF}"

for command_name in curl tar sbatch; do
  command -v "$command_name" >/dev/null || {
    echo "Missing required command: $command_name" >&2
    exit 2
  }
done

mkdir -p "$PROJECT_ROOT" "$TARGET"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

curl -fsSL "$ARCHIVE_URL" -o "$TMP_DIR/repository.tar.gz"
tar -xzf "$TMP_DIR/repository.tar.gz" -C "$TMP_DIR"
SOURCE_DIR="$(find "$TMP_DIR" -type d -path '*/jfm_rt05_kn001_heavy_rerun' -print -quit)"
if [[ -z "$SOURCE_DIR" ]]; then
  echo "Rerun package not found in GitHub archive for ref: $REF" >&2
  exit 3
fi

# Preserve any existing run_output directory while refreshing versioned code.
cp -a "$SOURCE_DIR/." "$TARGET/"
cd "$TARGET"
bash scripts/submit.sh
