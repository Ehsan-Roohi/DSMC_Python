#!/usr/bin/env bash
#SBATCH --job-name=r26-mm-k005
#SBATCH --partition=cpu
#SBATCH --qos=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-r26-mm-k005-%j.out
#SBATCH --error=slurm-r26-mm-k005-%j.err

set -euo pipefail
BUNDLE_ROOT=${R26_MAXWELL_BUNDLE_ROOT:?set by submit_maxwell_pair.sh}
RESULT_ROOT=${R26_MAXWELL_RESULT_ROOT:?set by submit_maxwell_pair.sh}
SEED=${R26_MAXWELL_SEED_KN005:?path to accepted KnGu=0.05 N40 state used only as an initial guess}
CODE="$BUNDLE_ROOT/source/r26/code"
DRIVER="$BUNDLE_ROOT/source/r26/analysis/run_jfm_observability_continuation.py"
VALIDATE="$BUNDLE_ROOT/tests/validate_r26_maxwell_run.py"
OUT="$RESULT_ROOT/kn005_N40"
[[ -s "$SEED" ]] || { echo "missing seed: $SEED" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite $OUT" >&2; exit 2; }

export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

python3 "$CODE/run_tests.py"
python3 "$BUNDLE_ROOT/tests/test_maxwell_contract.py"
python3 "$DRIVER" --nodes 40 --case-family jfm-maxwell --kn-gu 0.05 \
  --lid-speed-m-s 100 --wall-temperature-k 300 --beta 1.25 \
  --closure-mode jfm2009 --initial-state "$SEED" --reconcile-initial \
  --solver colored_newton --raw-tolerance 1e-8 --solver-tolerance 1e-9 \
  --max-nfev 140 --output-dir "$OUT"
python3 "$VALIDATE" "$OUT" --expected-kn 0.05 --expected-nodes 40
sha256sum "$OUT/last_accepted_state.npz" "$OUT/run_summary.json" >"$OUT/FILES.sha256"
