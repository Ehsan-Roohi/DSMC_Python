#!/usr/bin/env bash
#SBATCH --job-name=moh_mv7_post
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${MV7_REPO_ROOT:?exported by submit script}"
MV6_ROOT="${MV7_MV6_ROOT:?exported by submit script}"
OUTPUT_ROOT="${MV7_OUTPUT_ROOT:?exported by submit script}"
VENV_DIR="${MV7_VENV_DIR:?exported by submit script}"
BASELINE_JOB_ID="${MV7_BASELINE_JOB_ID:?exported by submit script}"
MODEL_JOB_ID="${MV7_MODEL_JOB_ID:?exported by submit script}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

ACCOUNTING_IDS="${BASELINE_JOB_ID},${MODEL_JOB_ID}"
if [[ -n "${MV7_SOURCE_MV6_MODEL_JOB_ID:-}" ]]; then
  ACCOUNTING_IDS="${ACCOUNTING_IDS},${MV7_SOURCE_MV6_MODEL_JOB_ID}"
fi
if [[ -n "${MV7_SOURCE_REFERENCE_JOB_ID:-}" ]]; then
  ACCOUNTING_IDS="${ACCOUNTING_IDS},${MV7_SOURCE_REFERENCE_JOB_ID}"
fi

if ! sacct -j "${ACCOUNTING_IDS}" -X -P \
  --format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,CPUTimeRAW,MaxRSS,NodeList \
  > "${OUTPUT_ROOT}/slurm_accounting.psv"; then
  printf 'JobIDRaw|JobName|State|ExitCode|ElapsedRaw|CPUTimeRAW|MaxRSS|NodeList|\n' \
    > "${OUTPUT_ROOT}/slurm_accounting.psv"
fi

cd "${REPO_ROOT}"
source "${VENV_DIR}/bin/activate"
python -m vgdsmc.mohammadzadeh_mv7_jcp_budget_matrix \
  --mode post \
  --mv6-root "${MV6_ROOT}" \
  --output-root "${OUTPUT_ROOT}"
