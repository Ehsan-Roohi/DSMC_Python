#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN=${CAMPAIGN:-/project/pi_roohie_umass_edu/CavityColdToHotIdentify/JFM_Five_Run_Campaign_20260802}
R13_N60_RESULT=${R13_N60_RESULT:-$CAMPAIGN/results/run6_kn020_recovery_v2/run_20260806T014740Z/r13_fast_target/N60_20260810T224553Z/result}
PARTITION=${PARTITION:-cpu}
CPUS=${CPUS:-4}
MEMORY=${MEMORY:-32G}
WALLTIME=${WALLTIME:-1-00:00:00}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
SOURCE_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)
SOURCE_SHORT=$(git -C "$REPO_ROOT" rev-parse --short=12 HEAD)
SOLVER_SOURCE=$REPO_ROOT/r13_recovery/code/rana_original_reference_solver.py
TRANSFER_SOURCE=$REPO_ROOT/r13_recovery/code/r13_grid_transfer.py
COEFFICIENT_SOURCE=$CAMPAIGN/code/r13/rana_original_coefficients.py
N60_STATE=$R13_N60_RESULT/state.npy
N60_REPORT=$R13_N60_RESULT/report.json

for required in \
  "$SOLVER_SOURCE" \
  "$TRANSFER_SOURCE" \
  "$COEFFICIENT_SOURCE" \
  "$N60_STATE" \
  "$N60_REPORT"; do
  test -f "$required" || {
    echo "Missing required file: $required" >&2
    exit 1
  }
done

python3 - "$N60_STATE" "$N60_REPORT" "$COEFFICIENT_SOURCE" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

state, report_path, coefficients = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text())

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

expected_state = report["execution_provenance"]["output_state_sha256"]
expected_coefficients = report["execution_provenance"][
    "coefficient_source_sha256_at_end"
]
assert digest(state) == expected_state, "N60 state/report hash mismatch"
assert digest(coefficients) == expected_coefficients, "R13 coefficient hash mismatch"
assert report["passed_private_run_gates"] is True
assert report["metrics_valid"] is True
assert report["state_semantics"] == "accepted_private_physical_solution"
assert report["configuration"]["nx"] == 60
assert report["configuration"]["ny"] == 60
assert abs(report["configuration"]["kn"] - 0.159576912160573) < 1.0e-14
print("R13_N60_SOURCE_AND_COEFFICIENTS_VERIFIED")
PY

RUN_ROOT=$CAMPAIGN/results/run6_kn020_recovery_v3/r13_n80_kn020_$(date -u +%Y%m%dT%H%M%SZ)
RUNTIME=$RUN_ROOT/runtime
RESULT=$RUN_ROOT/result
mkdir -p "$RUNTIME"
cp "$SOLVER_SOURCE" "$RUNTIME/rana_original_reference_solver.py"
cp "$TRANSFER_SOURCE" "$RUNTIME/r13_grid_transfer.py"
cp "$COEFFICIENT_SOURCE" "$RUNTIME/rana_original_coefficients.py"

cat > "$RUN_ROOT/run_n80.sbatch" <<SBATCH
#!/usr/bin/env bash
#SBATCH --job-name=r13-n80-jfnk
#SBATCH --partition=$PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEMORY
#SBATCH --time=$WALLTIME
#SBATCH --output=$RUN_ROOT/slurm-%j.out
#SBATCH --error=$RUN_ROOT/slurm-%j.err

set -euo pipefail
cd "$RUNTIME"
export OMP_NUM_THREADS=$CPUS
export OPENBLAS_NUM_THREADS=$CPUS
export MKL_NUM_THREADS=$CPUS
export PYTHONUNBUFFERED=1

python3 rana_original_reference_solver.py \
  --nx 80 \
  --ny 80 \
  --kn 0.159576912160573 \
  --lid-velocity 0.40019264077087846 \
  --rb 1.0 \
  --ra 1.0 \
  --ma 1.0 \
  --boundary-scheme paper-linear \
  --pressure-mode paper-tangential \
  --continuity-mode conservative-fv \
  --conservative-linearization defect-newton \
  --nonlinear-solver jfnk \
  --linear-solver direct \
  --outer-tolerance 1.0e-10 \
  --max-outer-iterations 40 \
  --outer-relaxation 1.0 \
  --outer-globalization residual-backtracking \
  --line-search-reduction 0.5 \
  --line-search-min-step 0.0009765625 \
  --line-search-armijo 0.0001 \
  --physical-floor 1.0e-12 \
  --jfnk-gmres-restart 40 \
  --jfnk-gmres-max-cycles 14 \
  --jfnk-initial-forcing 0.01 \
  --jfnk-min-forcing 0.000001 \
  --jfnk-max-forcing 0.1 \
  --initial-state "$N60_STATE" \
  --continuation-reason "same-Kn N60-to-N80 grid refinement from Git commit $SOURCE_COMMIT" \
  --output-dir "$RESULT"

jq -e '
  .passed_private_run_gates == true and
  .metrics_valid == true and
  .state_semantics == "accepted_private_physical_solution" and
  .provenance_gates.initial_state_traceable == true and
  .execution_provenance.initial_state.grid_transfer.applied == true and
  .configuration.nx == 80 and
  .configuration.ny == 80
' "$RESULT/report.json" >/dev/null

echo R13_N80_ACCEPTED
SBATCH

JOB_ID=$(sbatch --parsable "$RUN_ROOT/run_n80.sbatch")
cat > "$RUN_ROOT/N80_SUBMISSION.env" <<ENV
R13_N80_JOB=$JOB_ID
R13_N80_OUT=$RUN_ROOT
R13_N80_RESULT=$RESULT
R13_SOURCE_COMMIT=$SOURCE_COMMIT
R13_N60_RESULT=$R13_N60_RESULT
ENV

echo "R13_N80_JOB=$JOB_ID"
echo "R13_N80_OUT=$RUN_ROOT"
echo "R13_N80_RESULT=$RESULT"
echo "R13_SOURCE_COMMIT=$SOURCE_COMMIT"
squeue -j "$JOB_ID"
