#!/usr/bin/env bash
#SBATCH --job-name=jcp1
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

: "${JCP1_REPO_ROOT:?}"
: "${JCP1_MV15C_ROOT:?}"
: "${JCP1_WORK_ROOT:?}"
: "${JCP1_PYTHON:?}"
: "${JCP1_SCRIPT:?}"

export PYTHONPATH="${JCP1_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

cd "${JCP1_WORK_ROOT}"
"${JCP1_PYTHON}" "${JCP1_SCRIPT}" run \
  --repo-root "${JCP1_REPO_ROOT}" \
  --mv15c-root "${JCP1_MV15C_ROOT}" \
  --output-dir "${JCP1_WORK_ROOT}/out" \
  --archive "${JCP1_WORK_ROOT}/JCP1.zip" \
  --bin-width 10 \
  --batch-size 4 \
  > "${JCP1_WORK_ROOT}/run.json"

test -s "${JCP1_WORK_ROOT}/JCP1.zip"
test -s "${JCP1_WORK_ROOT}/JCP1.zip.sha256"
sha256sum -c "${JCP1_WORK_ROOT}/JCP1.zip.sha256"
touch "${JCP1_WORK_ROOT}/JCP1_DONE"
printf 'UPLOAD:\n%s\n%s\n' \
  "${JCP1_WORK_ROOT}/JCP1.zip" \
  "${JCP1_WORK_ROOT}/JCP1.zip.sha256"
