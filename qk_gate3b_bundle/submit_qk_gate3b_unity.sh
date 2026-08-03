#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/pi_roohie_umass_edu/Combustion/QK_GATE3B_LIVE
REPO="$ROOT/DSMC_Python"
BUNDLE="$ROOT/qk_gate3_source.tar.gz"
SRC="$ROOT/qk_gate3_shock_induction"
EXPECTED_SHA=ebe4ef97a2391a7b9c2aeb72a428e14f5bd4dccd8d0fdf1c1928766bef9696be

mkdir -p "$ROOT/runs"
if [[ -d "$REPO/.git" ]]; then
  git -C "$REPO" fetch --depth 1 origin main
  git -C "$REPO" reset --hard origin/main
else
  rm -rf "$REPO"
  git clone --depth 1 https://github.com/Ehsan-Roohi/DSMC_Python.git "$REPO"
fi

cat "$REPO"/qk_gate3_bundle/chunk_*.b64 \
  | tr -d '\n\r' \
  | base64 --decode > "$BUNDLE"
echo "$EXPECTED_SHA  $BUNDLE" | sha256sum -c -
rm -rf "$SRC"
tar -xzf "$BUNDLE" -C "$ROOT"

cp "$REPO/qk_gate3b_bundle/compare_qk_gate3b.py" "$SRC/"
cp "$REPO/qk_gate3b_bundle/run_qk_gate3b_validation.sh" "$SRC/"
cp "$REPO/qk_gate3b_bundle/run_qk_gate3b_unity.sbatch" "$SRC/"
cp "$REPO/qk_gate3b_bundle/status_qk_gate3b_unity.sh" "$SRC/"
chmod +x "$SRC"/*.sh "$SRC"/*.sbatch "$SRC"/compare_qk_gate3b.py

{
  echo "repository_commit=$(git -C "$REPO" rev-parse HEAD)"
  echo "base_gate3_bundle_sha256=$EXPECTED_SHA"
  echo "python_reference=live"
  echo "seed_start=58001"
  echo "seed_count=32"
  echo "crossing_method=linear_interpolation"
} > "$SRC/GATE3B_PROVENANCE.txt"

JOB_ID=$(sbatch --parsable \
  --output="$ROOT/runs/slurm-%j.out" \
  --error="$ROOT/runs/slurm-%j.err" \
  "$SRC/run_qk_gate3b_unity.sbatch")

cat > "$ROOT/LAST_GATE3B_JOB.env" <<EOT
JOB_ID=$JOB_ID
ROOT=$ROOT
OUT=$ROOT/runs/slurm-$JOB_ID.out
ERR=$ROOT/runs/slurm-$JOB_ID.err
RESULT_DIR=$ROOT/runs/job_$JOB_ID
EOT

echo "Submitted live 32-seed Bird-QK Gate 3B to UMass Unity."
echo "JOB_ID=$JOB_ID"
echo "Status: squeue -j $JOB_ID"
echo "Log: tail -f $ROOT/runs/slurm-$JOB_ID.out"
echo "Report: $ROOT/runs/job_$JOB_ID/validation_output/QK_GATE3B_VALIDATION_REPORT.txt"
