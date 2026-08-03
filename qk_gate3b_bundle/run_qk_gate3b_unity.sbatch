#!/usr/bin/env bash
#SBATCH --job-name=qk_gate3b
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=roohie@umass.edu

set -euo pipefail

ROOT=/project/pi_roohie_umass_edu/Combustion/QK_GATE3B_LIVE
SRC="$ROOT/qk_gate3_shock_induction"
RUN_DIR="$ROOT/runs/job_${SLURM_JOB_ID}"
mkdir -p "$ROOT/runs"
RUN_LOG="$ROOT/runs/job_${SLURM_JOB_ID}.unity_gate3b_full.log"

module purge >/dev/null 2>&1 || true
if ! command -v gfortran >/dev/null 2>&1; then
  module load gcc >/dev/null 2>&1 \
    || module load gcc/12.2.0 >/dev/null 2>&1 \
    || true
fi
command -v gfortran >/dev/null 2>&1 \
  || { echo 'ERROR: gfortran not found' >&2; exit 2; }

PYTHON_BIN=''
for candidate in \
  /home/roohie_umass_edu/miniconda3/bin/python \
  /work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python \
  "$(command -v python3 2>/dev/null || true)" \
  "$(command -v python 2>/dev/null || true)"; do
  if [[ -n "$candidate" && -x "$candidate" ]] \
     && "$candidate" -c 'import numpy, numba' >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done
[[ -n "$PYTHON_BIN" ]] \
  || { echo 'ERROR: no Python with numpy and numba found' >&2; exit 3; }

export NPROC="${SLURM_CPUS_PER_TASK:-8}"
export NSEEDS=32
export MIN_SEEDS=32
export SEED_START=58001
export PYTHON_BIN

bash "$SRC/run_qk_gate3b_validation.sh" "$RUN_DIR" 2>&1 | tee "$RUN_LOG"
cp "$RUN_LOG" "$RUN_DIR/unity_gate3b_full.log"

echo 'UNITY_QK_GATE3B_PASS'
echo "RESULT_DIR=$RUN_DIR"
echo "REPORT=$RUN_DIR/validation_output/QK_GATE3B_VALIDATION_REPORT.txt"
