#!/usr/bin/env bash
#SBATCH --job-name=qk_gate2
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=roohie@umass.edu

set -euo pipefail

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

ROOT=/project/pi_roohie_umass_edu/Combustion/QK_GATE2_UNITY
REPO="$ROOT/DSMC_Python"
RUN_DIR="$ROOT/runs/job_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

module purge >/dev/null 2>&1 || true
if ! command -v gfortran >/dev/null 2>&1; then
  module load gcc >/dev/null 2>&1 || module load gcc/12.2.0 >/dev/null 2>&1 || true
fi

if ! command -v gfortran >/dev/null 2>&1; then
  echo "ERROR: gfortran not found after module loading." >&2
  exit 2
fi

PYTHON=/home/roohie_umass_edu/miniconda3/bin/python
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3 || true)"
fi
if [[ -z "$PYTHON" ]]; then
  echo "ERROR: Python 3 not found." >&2
  exit 3
fi
if ! "$PYTHON" -c 'import numpy' >/dev/null 2>&1; then
  echo "ERROR: NumPy is not available in $PYTHON." >&2
  exit 4
fi

if [[ -d "$REPO/.git" ]]; then
  git -C "$REPO" fetch --depth 1 origin main
  git -C "$REPO" reset --hard origin/main
else
  rm -rf "$REPO"
  git clone --depth 1 https://github.com/Ehsan-Roohi/DSMC_Python.git "$REPO"
fi

cd "$REPO"
cat qk_gate2_bundle/chunk_*.b64 | tr -d '\n\r' | base64 --decode > "$RUN_DIR/qk_gate2_lean.tar.gz"
echo "c5e8c050ae9657ef5afffd218dbfef92bc6736e7a6ff03fee719f73ad7db1e98  $RUN_DIR/qk_gate2_lean.tar.gz" | sha256sum -c -

tar -xzf "$RUN_DIR/qk_gate2_lean.tar.gz" -C "$RUN_DIR"
cd "$RUN_DIR/qk_gate2"
chmod +x run_qk_gate2_validation.sh

# Ensure the validation scripts use the known Python interpreter.
export PATH="$(dirname "$PYTHON"):$PATH"

./run_qk_gate2_validation.sh 2>&1 | tee "$RUN_DIR/gate2_full.log"

cp -a validation_output "$RUN_DIR/"
cp -a fortran_runs "$RUN_DIR/"

REPORT="$RUN_DIR/validation_output/QK_GATE2_VALIDATION_REPORT.txt"
grep -q 'OVERALL=PASS' "$REPORT"
grep -q 'QK_GATE2_CROSS_LANGUAGE_GATE_PASS' "$RUN_DIR/validation_output/comparison_stdout.txt"

echo "UNITY_QK_GATE2_PASS"
echo "RESULT_DIR=$RUN_DIR"
echo "REPORT=$REPORT"
