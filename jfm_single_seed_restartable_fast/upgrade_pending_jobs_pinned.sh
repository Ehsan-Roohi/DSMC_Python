#!/bin/bash
set -euo pipefail

REPOSITORY="Ehsan-Roohi/DSMC_Python"
REF="${JFM_GITHUB_REF:-fd63ffe23c72a179b0b04864da37f1c855f9cb48}"
JOB_ID="${JFM_JOB_ID:-62675716}"
PROJECT_ROOT="${JFM_PROJECT_ROOT:-/project/pi_roohie_umass_edu/JFM_revision_2026}"
TARGET="${JFM_CHECKPOINT_FAST_ROOT:-$PROJECT_ROOT/JFM_SINGLE_SEED_CHECKPOINT_FAST}"
ARCHIVE_URL="https://codeload.github.com/${REPOSITORY}/tar.gz/${REF}"

for name in curl tar scontrol squeue python sha256sum; do
  command -v "$name" >/dev/null || { echo "Missing required command: $name" >&2; exit 2; }
done
test -d "$TARGET/solver"

if ! squeue -h -j "$JOB_ID" | grep -q .; then
  echo "Job $JOB_ID is not present in the current queue; no files changed" >&2
  exit 3
fi

echo "Holding pending tasks in job $JOB_ID during the atomic solver upgrade..."
scontrol hold "$JOB_ID"
HELD=1
release_on_exit() {
  if [[ "${HELD:-0}" == 1 ]]; then
    scontrol release "$JOB_ID" >/dev/null 2>&1 || true
  fi
}
trap release_on_exit EXIT

if squeue -h -j "$JOB_ID" -t R | grep -q .; then
  echo "At least one task in $JOB_ID is already running; upgrade aborted" >&2
  exit 4
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"; release_on_exit' EXIT
curl -fsSL "$ARCHIVE_URL" -o "$TMP_DIR/repository.tar.gz"
tar -xzf "$TMP_DIR/repository.tar.gz" -C "$TMP_DIR"
SOURCE="$(find "$TMP_DIR" -type d -path '*/jfm_single_seed_restartable_fast' -print -quit)"
[[ -n "$SOURCE" ]] || { echo "Restartable package not found at $REF" >&2; exit 5; }

python -m py_compile "$SOURCE"/solver/*.py
for script in "$SOURCE"/scripts/*.sh "$SOURCE"/scripts/*.slurm; do
  bash -n "$script"
done

HS_SHA="$(sha256sum "$SOURCE/solver/JFM_hs_dsmc_quarter.py" | awk '{print $1}')"
RELAX_SHA="$(sha256sum "$SOURCE/solver/JFM_bgk_shakhov_quarter.py" | awk '{print $1}')"
[[ "$HS_SHA" == "2c9e2f5119802b123f0335664a564085cfe77682ae4f4af4138dc31f5876b166" ]]
[[ "$RELAX_SHA" == "f2f97526942c53eca0af6dd9b94aca541f06951db4755a92cc8fb11e5e0e65a2" ]]

mkdir -p "$TARGET/solver" "$TARGET/scripts"
cp "$SOURCE/solver/JFM_hs_dsmc_quarter.py" \
  "$TARGET/solver/JFM_hs_dsmc_quarter.py.new"
cp "$SOURCE/solver/JFM_bgk_shakhov_quarter.py" \
  "$TARGET/solver/JFM_bgk_shakhov_quarter.py.new"
mv "$TARGET/solver/JFM_hs_dsmc_quarter.py.new" \
  "$TARGET/solver/JFM_hs_dsmc_quarter.py"
mv "$TARGET/solver/JFM_bgk_shakhov_quarter.py.new" \
  "$TARGET/solver/JFM_bgk_shakhov_quarter.py"
cp "$SOURCE/scripts/run_continuation.slurm" "$TARGET/scripts/"
cp "$SOURCE/scripts/summarize_continuation.slurm" "$TARGET/scripts/"
cp "$SOURCE/scripts/submit_continuation.sh" "$TARGET/scripts/"
cp "$SOURCE/SOURCE_PROVENANCE.md" "$TARGET/"

python -m py_compile "$TARGET"/solver/*.py
[[ "$(sha256sum "$TARGET/solver/JFM_hs_dsmc_quarter.py" | awk '{print $1}')" == "$HS_SHA" ]]
[[ "$(sha256sum "$TARGET/solver/JFM_bgk_shakhov_quarter.py" | awk '{print $1}')" == "$RELAX_SHA" ]]

HELD=0
scontrol release "$JOB_ID"
echo "[OK] Job $JOB_ID released with restartable solvers"
echo "HS_SHA256=$HS_SHA"
echo "BGK_SHAKHOV_SHA256=$RELAX_SHA"
echo "No job was cancelled or resubmitted."
