#!/bin/bash
#SBATCH --job-name=r26-r13resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@300

set -euo pipefail
: "${R26_SOURCE_DIR:?}"
: "${R26_RUN_BASE:?}"
: "${R26_CAMPAIGN:?}"
: "${R26_PYTHON:?}"
: "${R26_R13_RESULT:?}"
: "${R26_RESUME_BASE:?}"
: "${R26_REFINE_NODES:?}"

ROUTES=(r13_full_uniform r13_quarter_stretched)
ROUTE=${ROUTES[$SLURM_ARRAY_TASK_ID]}
OUT=$R26_RUN_BASE/$ROUTE
ZIP=$R26_CAMPAIGN/R26_R13RESUME_${ROUTE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.zip
PACKAGED=0

package_output () {
    if test "$PACKAGED" -eq 1 || test ! -d "$OUT"; then return; fi
    PACKAGED=1
    "$R26_PYTHON" - "$OUT" "$ZIP" <<'PY'
from pathlib import Path
import shutil
import sys
source = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
target.parent.mkdir(parents=True, exist_ok=True)
temporary = str(target.with_suffix("")) + ".partial"
archive = Path(shutil.make_archive(temporary, "zip", source.parent, source.name))
archive.replace(target)
print(f"ZIP={target}")
PY
}

on_term () { package_output; exit 143; }
trap on_term TERM
trap package_output EXIT

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

"$R26_PYTHON" "$R26_SOURCE_DIR/r26_recovery/analysis/run_r26_r13_seeded_route.py" \
    --route "$ROUTE" \
    --repo-root "$R26_SOURCE_DIR" \
    --r13-result "$R26_R13_RESULT" \
    --resume-base "$R26_RESUME_BASE" \
    --output-dir "$OUT" \
    --winner-dir "$R26_RUN_BASE/winner" \
    --refine-nodes "${R26_REFINE_NODES//:/,}" \
    --continuation-timeout 43200
