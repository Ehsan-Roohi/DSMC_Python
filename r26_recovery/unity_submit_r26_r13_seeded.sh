#!/bin/bash
# Submit three fail-closed R13-seeded R26 recovery routes on Unity.

set -euo pipefail

CAMPAIGN=${CAMPAIGN:-/project/pi_roohie_umass_edu/CavityColdToHotIdentify/JFM_Five_Run_Campaign_20260802}
REPOSITORY=${REPOSITORY:-https://github.com/Ehsan-Roohi/DSMC_Python.git}
SOURCE_REF=${SOURCE_REF:-agent/r26-r13-seeded-recovery-20260815}
R26_PYTHON=${R26_PYTHON:-python3}
R13_RESULT=${R13_RESULT:-$CAMPAIGN/results/run6_kn020_recovery_v2/run_20260806T014740Z/r13_fast_target/N60_20260810T224553Z/result}
PARTITION=${PARTITION:-cpu}
CPUS=${CPUS:-4}
MEMORY=${MEMORY:-32G}
WALLTIME=${WALLTIME:-3-00:00:00}
REFINE_NODES=${REFINE_NODES:-16,20,24,28,32,36,40}

test -d "$CAMPAIGN"
test -f "$R13_RESULT/state.npy"
test -f "$R13_RESULT/report.json"
command -v git >/dev/null
command -v sbatch >/dev/null
"$R26_PYTHON" -c 'import numpy, scipy; print("numpy", numpy.__version__, "scipy", scipy.__version__)'

SAFE_REF=${SOURCE_REF//[^A-Za-z0-9]/_}
SOURCE_DIR=$CAMPAIGN/software/DSMC_Python_r26_r13seed_${SAFE_REF:0:48}
if test ! -d "$SOURCE_DIR/.git"; then
    mkdir -p "$(dirname "$SOURCE_DIR")"
    git clone --filter=blob:none --no-checkout "$REPOSITORY" "$SOURCE_DIR"
    git -C "$SOURCE_DIR" sparse-checkout init --cone
    git -C "$SOURCE_DIR" sparse-checkout set r13_recovery r26_recovery
fi
git -C "$SOURCE_DIR" fetch --depth=1 origin "$SOURCE_REF"
git -C "$SOURCE_DIR" checkout --detach FETCH_HEAD
SOURCE_SHA=$(git -C "$SOURCE_DIR" rev-parse HEAD)

"$R26_PYTHON" -m py_compile \
    "$SOURCE_DIR/r26_recovery/analysis/run_r26_r13_seeded_route.py" \
    "$SOURCE_DIR/r26_recovery/analysis/run_jfm_observability_continuation.py" \
    "$SOURCE_DIR/r26_recovery/code/"r26_*.py
bash -n "$SOURCE_DIR/r26_recovery/r26_r13_seeded_route.slurm"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_BASE=$CAMPAIGN/results/run6_kn020_recovery_v5/r26_r13seed_${STAMP}_${SOURCE_SHA:0:12}
mkdir -p "$RUN_BASE"
printf '%s\n' "$SOURCE_SHA" > "$RUN_BASE/SOURCE_COMMIT.txt"

ARRAY_JOB=$(sbatch --parsable \
    --partition="$PARTITION" \
    --array=0-2%3 \
    --cpus-per-task="$CPUS" \
    --mem="$MEMORY" \
    --time="$WALLTIME" \
    --output="$RUN_BASE/slurm-%A_%a.out" \
    --error="$RUN_BASE/slurm-%A_%a.err" \
    --export="ALL,R26_SOURCE_DIR=$SOURCE_DIR,R26_RUN_BASE=$RUN_BASE,R26_CAMPAIGN=$CAMPAIGN,R26_PYTHON=$R26_PYTHON,R26_R13_RESULT=$R13_RESULT,R26_REFINE_NODES=${REFINE_NODES//,/:}" \
    "$SOURCE_DIR/r26_recovery/r26_r13_seeded_route.slurm")

printf '%s\n' \
    "SOURCE_SHA=$SOURCE_SHA" \
    "SOURCE_DIR=$SOURCE_DIR" \
    "RUN_BASE=$RUN_BASE" \
    "ARRAY_JOB=$ARRAY_JOB" \
    "R13_RESULT=$R13_RESULT" \
    "ROUTES=r13_full_uniform,r13_half_stretched,r13_quarter_stretched" \
    > "$RUN_BASE/submission.env"

echo "SOURCE_SHA=$SOURCE_SHA"
echo "RUN_BASE=$RUN_BASE"
echo "ARRAY_JOB=$ARRAY_JOB"
echo "Submitted: 0=r13_full_uniform 1=r13_half_stretched 2=r13_quarter_stretched"
echo "ZIP outputs will be written directly under $CAMPAIGN"
