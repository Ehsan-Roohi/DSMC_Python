#!/bin/bash
set -euo pipefail

MODE="${1:-}"
case "$MODE" in
  vram48|vram80) ;;
  *) echo "Usage: $0 {vram48|vram80}" >&2; exit 2 ;;
esac

REPOSITORY="Ehsan-Roohi/DSMC_Python"
REF="${JFM_GITHUB_REF:-b243a7ac1878eafd6ece51a7553a631b50a2e389}"
PROJECT_ROOT="${JFM_PROJECT_ROOT:-/project/pi_roohie_umass_edu/JFM_revision_2026}"
TARGET="${JFM_HIGHSTAT_ROOT:-$PROJECT_ROOT/JFM_HIGHSTAT_FIGURES_80M_S100K}"
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
SOURCE_DIR="$(find "$TMP_DIR" -type d -path '*/jfm_highstat_figures' -print -quit)"
if [[ -z "$SOURCE_DIR" ]]; then
  echo "High-statistics package not found for ref: $REF" >&2
  exit 4
fi

cp -a "$SOURCE_DIR/." "$TARGET/"
cd "$TARGET"
bash scripts/submit_mode.sh "$MODE"
