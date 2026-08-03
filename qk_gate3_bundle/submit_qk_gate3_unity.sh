#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/pi_roohie_umass_edu/Combustion/QK_GATE3_UNITY
REPO="$ROOT/DSMC_Python"
BUNDLE="$ROOT/qk_gate3_source.tar.gz"
SRC="$ROOT/qk_gate3_shock_induction"
EXPECTED_SHA=2afabd49087a0bb3ebbf1b9ed014d5b4b4c672f2c1c9f5701232d9d7301639ce

mkdir -p "$ROOT/runs"
if [[ -d "$REPO/.git" ]]; then
  git -C "$REPO" fetch --depth 1 origin main
  git -C "$REPO" reset --hard origin/main
else
  rm -rf "$REPO"
  git clone --depth 1 https://github.com/Ehsan-Roohi/DSMC_Python.git "$REPO"
fi

cat "$REPO"/qk_gate3_bundle/chunk_*.b64 | tr -d '\n\r' | base64 --decode > "$BUNDLE"
echo "$EXPECTED_SHA  $BUNDLE" | sha256sum -c -
rm -rf "$SRC"
tar -xzf "$BUNDLE" -C "$ROOT"
chmod +x "$SRC"/*.sh "$SRC"/*.sbatch

JOB_ID=$(sbatch --parsable \
  --output="$ROOT/runs/slurm-%j.out" \
  --error="$ROOT/runs/slurm-%j.err" \
  "$SRC/run_qk_gate3_unity.sbatch")

cat > "$ROOT/LAST_GATE3_JOB.env" <<EOT
JOB_ID=$JOB_ID
ROOT=$ROOT
OUT=$ROOT/runs/slurm-$JOB_ID.out
ERR=$ROOT/runs/slurm-$JOB_ID.err
RESULT_DIR=$ROOT/runs/job_$JOB_ID
EOT

echo "Submitted Bird-QK Gate 3 to UMass Unity."
echo "JOB_ID=$JOB_ID"
echo "Status: squeue -j $JOB_ID"
echo "Log: tail -f $ROOT/runs/slurm-$JOB_ID.out"
echo "Report: $ROOT/runs/job_$JOB_ID/validation_output/QK_GATE3_VALIDATION_REPORT.txt"
