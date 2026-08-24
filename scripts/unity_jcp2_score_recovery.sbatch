#!/usr/bin/env bash
#SBATCH --job-name=j2-score-e
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

: "${JCP2_REPO_ROOT:?}"
: "${JCP2_RUN_ROOT:?}"
: "${JCP2_PREDICTION_ROOT:?}"
: "${JCP2_SCORE_ROOT:?}"
: "${JCP2_WORK_ROOT:?}"
: "${JCP2_PYTHON:?}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${JCP2_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export JCP2_SELECTION_AUDIT_ROOT="${JCP2_WORK_ROOT}"

cd "${JCP2_REPO_ROOT}"
"${JCP2_PYTHON}" -m vgdsmc.jcp_phase1_recovery score \
  --run-root "${JCP2_RUN_ROOT}" \
  --prediction-dir "${JCP2_PREDICTION_ROOT}" \
  --output-dir "${JCP2_SCORE_ROOT}" \
  --archive "${JCP2_WORK_ROOT}/JCP2.zip"

test -s "${JCP2_WORK_ROOT}/evaluation_selection_audit.json"
test -s "${JCP2_WORK_ROOT}/reference_selection_audit.json"
test -s "${JCP2_WORK_ROOT}/JCP2.zip"
test -s "${JCP2_WORK_ROOT}/JCP2.zip.sha256"
cd "${JCP2_WORK_ROOT}"
sha256sum -c JCP2.zip.sha256
touch JCP2_DONE
