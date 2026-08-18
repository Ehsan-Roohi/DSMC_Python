#!/bin/bash
set -euo pipefail

REPOSITORY="Ehsan-Roohi/DSMC_Python"
REF="${JFM_GITHUB_REF:-58e03f511b5dde628a5e15cfe923a4690b6062ce}"
PROJECT_ROOT="${JFM_PROJECT_ROOT:-/project/pi_roohie_umass_edu/JFM_revision_2026}"
TARGET="${JFM_CHECKPOINT_FAST_ROOT:-$PROJECT_ROOT/JFM_SINGLE_SEED_CHECKPOINT_FAST}"
ARCHIVE_URL="https://codeload.github.com/${REPOSITORY}/tar.gz/${REF}"

for name in curl tar sbatch scancel; do
  command -v "$name" >/dev/null || { echo "Missing required command: $name" >&2; exit 2; }
done

mkdir -p "$PROJECT_ROOT" "$TARGET"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
curl -fsSL "$ARCHIVE_URL" -o "$TMP_DIR/repository.tar.gz"
tar -xzf "$TMP_DIR/repository.tar.gz" -C "$TMP_DIR"
SOURCE_DIR="$(find "$TMP_DIR" -type d -path '*/jfm_single_seed_checkpoint_fast' -print -quit)"
[[ -n "$SOURCE_DIR" ]] || { echo "Package not found at ref $REF" >&2; exit 4; }
cp -a "$SOURCE_DIR/." "$TARGET/"
cd "$TARGET"
bash scripts/submit_checkpoint_fast.sh "${1:-}"
