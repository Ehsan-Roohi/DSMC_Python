#!/usr/bin/env bash
set -euo pipefail

ROOT=/project/pi_roohie_umass_edu/Combustion/QK_GATE2_UNITY
REPO="$ROOT/DSMC_Python"
mkdir -p "$ROOT/runs"

if [[ -d "$REPO/.git" ]]; then
  git -C "$REPO" fetch --depth 1 origin main
  git -C "$REPO" reset --hard origin/main
else
  rm -rf "$REPO"
  git clone --depth 1 https://github.com/Ehsan-Roohi/DSMC_Python.git "$REPO"
fi

cd "$REPO"
JOB_ID=$(sbatch --parsable \
  --output="$ROOT/runs/slurm-%j.out" \
  --error="$ROOT/runs/slurm-%j.err" \
  unity_gate2/run_gate2_unity.sbatch)

cat > "$ROOT/LAST_GATE2_JOB.env" <<EOF
JOB_ID=$JOB_ID
ROOT=$ROOT
OUT=$ROOT/runs/slurm-$JOB_ID.out
ERR=$ROOT/runs/slurm-$JOB_ID.err
RESULT_DIR=$ROOT/runs/job_$JOB_ID
EOF

echo "Submitted Bird-QK Gate 2 to UMass Unity."
echo "JOB_ID=$JOB_ID"
echo "Status: squeue -j $JOB_ID"
echo "Log:    tail -f $ROOT/runs/slurm-$JOB_ID.out"
echo "Result: $ROOT/runs/job_$JOB_ID"
