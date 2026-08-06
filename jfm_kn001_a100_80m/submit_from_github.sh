#!/bin/bash
set -euo pipefail

REPOSITORY="Ehsan-Roohi/DSMC_Python"
# Immutable validated snapshot containing the solver, tests, and Slurm files.
REF="${JFM_GITHUB_REF:-f41d0076da72acedab9237e403f7160f3c6e3335}"
PROJECT_ROOT="${JFM_PROJECT_ROOT:-/project/pi_roohie_umass_edu/JFM_revision_2026}"
TARGET="${JFM_KN001_A100_ROOT:-$PROJECT_ROOT/JFM_RT05_KN001_A100_80M_S100K}"
ARCHIVE_URL="https://codeload.github.com/${REPOSITORY}/tar.gz/${REF}"

for command_name in curl tar sbatch; do
  command -v "$command_name" >/dev/null || { echo "Missing: $command_name" >&2; exit 2; }
done

mkdir -p "$PROJECT_ROOT" "$TARGET"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
curl -fsSL "$ARCHIVE_URL" -o "$TMP_DIR/repository.tar.gz"
tar -xzf "$TMP_DIR/repository.tar.gz" -C "$TMP_DIR"
SOURCE_DIR="$(find "$TMP_DIR" -type d -path '*/jfm_kn001_a100_80m' -print -quit)"
if [[ -z "$SOURCE_DIR" ]]; then
  echo "A100 Kn=0.01 package not found for ref: $REF" >&2
  exit 3
fi

cp -a "$SOURCE_DIR/." "$TARGET/"
cd "$TARGET"
bash scripts/submit.sh
