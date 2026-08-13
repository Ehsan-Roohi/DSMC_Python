#!/bin/bash
# Submit three independent, fail-closed R26 recovery routes on Unity.

set -euo pipefail

CAMPAIGN=${CAMPAIGN:-/project/pi_roohie_umass_edu/CavityColdToHotIdentify/JFM_Five_Run_Campaign_20260802}
REPOSITORY=${REPOSITORY:-https://github.com/Ehsan-Roohi/DSMC_Python.git}
SOURCE_REF=${SOURCE_REF:-agent/r26-kn020-recovery-ensemble-20260813}
R26_PYTHON=${R26_PYTHON:-python3}
PARTITION=${PARTITION:-cpu}
CPUS=${CPUS:-4}
MEMORY=${MEMORY:-48G}
WALLTIME=${WALLTIME:-4-00:00:00}
REFINE_NODES=${REFINE_NODES:-24,28,32,36,40}
REFINE_NODES_SLURM=${REFINE_NODES//,/:}

test -d "$CAMPAIGN"
command -v git >/dev/null
command -v sbatch >/dev/null
"$R26_PYTHON" -c 'import numpy, scipy; print("numpy", numpy.__version__, "scipy", scipy.__version__)'

SAFE_REF=${SOURCE_REF//[^A-Za-z0-9]/_}
SOURCE_DIR=$CAMPAIGN/software/DSMC_Python_r26_kn020_${SAFE_REF:0:48}
if test ! -d "$SOURCE_DIR/.git"; then
    mkdir -p "$(dirname "$SOURCE_DIR")"
    git clone --filter=blob:none --no-checkout "$REPOSITORY" "$SOURCE_DIR"
    git -C "$SOURCE_DIR" sparse-checkout init --cone
    git -C "$SOURCE_DIR" sparse-checkout set r26_recovery
fi
git -C "$SOURCE_DIR" fetch --depth=1 origin "$SOURCE_REF"
git -C "$SOURCE_DIR" checkout --detach FETCH_HEAD
SOURCE_SHA=$(git -C "$SOURCE_DIR" rev-parse HEAD)

"$R26_PYTHON" -m py_compile \
    "$SOURCE_DIR/r26_recovery/analysis/run_r26_kn020_route.py" \
    "$SOURCE_DIR/r26_recovery/analysis/run_jfm_observability_continuation.py" \
    "$SOURCE_DIR/r26_recovery/code/"r26_*.py

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_BASE=$CAMPAIGN/results/run6_kn020_recovery_v4/r26_kn020_ensemble_${STAMP}_${SOURCE_SHA:0:12}
mkdir -p "$RUN_BASE"
printf '%s\n' "$SOURCE_SHA" > "$RUN_BASE/SOURCE_COMMIT.txt"

SLURM_SCRIPT=$SOURCE_DIR/r26_recovery/r26_kn020_route.slurm
test -f "$SLURM_SCRIPT"

ARRAY_JOB=$(sbatch --parsable \
    --partition="$PARTITION" \
    --array=0-2%3 \
    --cpus-per-task="$CPUS" \
    --mem="$MEMORY" \
    --time="$WALLTIME" \
    --output="$RUN_BASE/slurm-%A_%a.out" \
    --error="$RUN_BASE/slurm-%A_%a.err" \
    --export="ALL,R26_SOURCE_DIR=$SOURCE_DIR,R26_RUN_BASE=$RUN_BASE,R26_CAMPAIGN=$CAMPAIGN,R26_PYTHON=$R26_PYTHON,R26_REFINE_NODES=$REFINE_NODES_SLURM" \
    "$SLURM_SCRIPT")

printf '%s\n' \
    "SOURCE_SHA=$SOURCE_SHA" \
    "SOURCE_DIR=$SOURCE_DIR" \
    "RUN_BASE=$RUN_BASE" \
    "ARRAY_JOB=$ARRAY_JOB" \
    "ROUTES=direct_colored,direct_trf,kn_ladder" \
    "REFINE_NODES=$REFINE_NODES" \
    > "$RUN_BASE/submission.env"

echo "SOURCE_SHA=$SOURCE_SHA"
echo "RUN_BASE=$RUN_BASE"
echo "ARRAY_JOB=$ARRAY_JOB"
echo "Submitted routes: 0=direct_colored 1=direct_trf 2=kn_ladder"
echo "Monitor: squeue -j $ARRAY_JOB; find '$RUN_BASE' -name route_summary.json -print -exec jq '{route,status,target_kn_gu,records:(.records|length)}' {} \;"
