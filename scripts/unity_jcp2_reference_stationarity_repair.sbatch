#!/usr/bin/env bash
#SBATCH --job-name=j2-ref-qc
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

: "${JCP2_REPO_ROOT:?}"
: "${JCP2_RUN_ROOT:?}"
: "${JCP2_PREDICTION_ROOT:?}"
: "${JCP2_REFERENCE_REPAIR_AUDIT:?}"
: "${JCP2_PYTHON:?}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${JCP2_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${JCP2_REPO_ROOT}"
"${JCP2_PYTHON}" - "${JCP2_PREDICTION_ROOT}" <<'PY'
import sys
from pathlib import Path

from vgdsmc import jcp_phase1_cavity as jcp2

root = Path(sys.argv[1])
prediction = root / "predictions.npz"
lock_path = root / "prediction_lock.json"
if not prediction.is_file() or not lock_path.is_file():
    raise FileNotFoundError("locked JCP2 prediction is absent")
lock = jcp2._json(lock_path)
if lock.get("status") != "predictions_locked_before_reference_interface":
    raise ValueError("JCP2 prediction lock has an invalid status")
if jcp2._sha256(prediction) != lock.get("prediction_sha256"):
    raise ValueError("JCP2 prediction hash differs from its pre-reference lock")
print("JCP2_PREDICTION_LOCK_VERIFIED_BEFORE_REFERENCE_REPAIR=1")
PY

"${JCP2_PYTHON}" -m vgdsmc.jcp_phase1_stationarity_repair \
  --run-root "${JCP2_RUN_ROOT}" \
  --group reference \
  --audit "${JCP2_REFERENCE_REPAIR_AUDIT}" \
  --apply

test -s "${JCP2_REFERENCE_REPAIR_AUDIT}"
