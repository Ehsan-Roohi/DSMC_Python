#!/bin/bash
set -euo pipefail

REPOSITORY="Ehsan-Roohi/DSMC_Python"
REF="${JFM_GITHUB_REF:-f97b3d6a8e97e2bc41ad1444f8d2c32e9e5b152f}"
PROJECT_ROOT="${JFM_PROJECT_ROOT:-/project/pi_roohie_umass_edu/JFM_revision_2026}"
TARGET="${JFM_A100_FAST_ROOT:-$PROJECT_ROOT/JFM_SINGLE_SEED_RESTARTABLE_A100_80G}"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$TARGET/run_output}"
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
bash scripts/preflight_local.sh

CASE_TABLE="$TARGET/cases/fast7.csv"
mkdir -p "$OUTPUT_ROOT" "$TARGET/slurm"
COMMON_EXPORT="ALL,JFM_ROOT=$TARGET,JFM_OUTPUT_ROOT=$OUTPUT_ROOT,JFM_CASE_TABLE=$CASE_TABLE"

RUN_JOB="$(sbatch --parsable \
  --job-name=jfm-a100r \
  --partition=gpu --gpus=1 --constraint=a100-80g \
  --cpus-per-task=4 --mem=24G --time=168:00:00 --array=0-6%4 \
  --output="$TARGET/slurm/jfm-a100r-%A_%a.out" \
  --error="$TARGET/slurm/jfm-a100r-%A_%a.err" \
  --export="$COMMON_EXPORT" scripts/run_checkpoint_fast.slurm)"

SUMMARY_JOB="$(sbatch --parsable \
  --dependency="afterok:$RUN_JOB" --kill-on-invalid-dep=yes \
  --job-name=jfm-a100r-sum \
  --partition=cpu --cpus-per-task=4 --mem=24G --time=08:00:00 \
  --output="$TARGET/slurm/jfm-a100r-summary-%j.out" \
  --error="$TARGET/slurm/jfm-a100r-summary-%j.err" \
  --export="$COMMON_EXPORT" scripts/summarize_checkpoint.slurm)"

printf 'RUN_JOB=%q\nSUMMARY_JOB=%q\nOUTPUT_ROOT=%q\nGPU_CONSTRAINT=%q\n' \
  "$RUN_JOB" "$SUMMARY_JOB" "$OUTPUT_ROOT" "a100-80g" \
  > "$TARGET/LAST_SUBMISSION.env"

echo "GPU_CONSTRAINT=a100-80g"
echo "RUN_JOB=$RUN_JOB"
echo "SUMMARY_JOB=$SUMMARY_JOB"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "Monitor: squeue -j $RUN_JOB,$SUMMARY_JOB"
echo "[OK] Independent A100-80GB restartable race submitted"
echo "Existing jobs were not cancelled or modified."
