#!/usr/bin/env bash
#SBATCH --job-name=r13-mm-jfm
#SBATCH --partition=cpu
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=slurm-r13-mm-%A_%a.out
#SBATCH --error=slurm-r13-mm-%A_%a.err

set -euo pipefail
: "${MAXWELL_MODEL_BUNDLE_ROOT:?required}"
: "${MAXWELL_MODEL_RESULT_ROOT:?required}"
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
R13="$MAXWELL_MODEL_BUNDLE_ROOT/r13"
SOURCE="$R13/source"
TESTS="$R13/tests"

case "$TASK" in
  0)
    KN_GU=0.05
    KN_RANA=0.039894228040143274
    NODES=60
    SEED="$R13/seeds/kn005_N60/state.npy"
    TAG=kn005_N60
    ;;
  1)
    KN_GU=0.20
    KN_RANA=0.1595769121605731
    NODES=60
    SEED="$R13/seeds/kn020_N60/state.npy"
    TAG=kn020_N60
    ;;
  *) echo "invalid R13 task: $TASK" >&2; exit 2 ;;
esac

OUT="$MAXWELL_MODEL_RESULT_ROOT/r13/$TAG"
[[ -s "$SEED" ]] || { echo "missing R13 initial state: $SEED" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")"
export PYTHONPATH="$SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

python3 "$TESTS/test_r13_maxwell_contract.py"
python3 "$SOURCE/rana_original_reference_solver.py" \
  --nx "$NODES" --ny "$NODES" --kn "$KN_RANA" \
  --lid-velocity 0.40019264077087846 \
  --rb 1 --ra 1 --ma 1 \
  --boundary-scheme paper-linear --pressure-mode paper-tangential \
  --continuity-mode conservative-fv --conservative-linearization defect-newton \
  --nonlinear-solver jfnk --linear-solver direct \
  --outer-tolerance 1e-10 --max-outer-iterations 50 \
  --outer-globalization residual-backtracking \
  --line-search-min-step 0.0009765625 \
  --jfnk-gmres-max-cycles 12 \
  --initial-state "$SEED" \
  --continuation-reason "reconcile archived state under Appendix-A Maxwell production at Kn_Gu=$KN_GU" \
  --output-dir "$OUT"
python3 "$TESTS/validate_r13_maxwell_run.py" "$OUT" \
  --expected-kn-rana "$KN_RANA" --expected-nodes "$NODES" | tee "$OUT/VALIDATION.txt"
sha256sum "$OUT/state.npy" "$OUT/report.json" > "$OUT/FILES.sha256"
