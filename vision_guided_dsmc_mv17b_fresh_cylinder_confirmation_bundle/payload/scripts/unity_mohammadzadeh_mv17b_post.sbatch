#!/usr/bin/env bash
#SBATCH --job-name=moh-mv17b-post
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00

set -euo pipefail
: "${MV17B_PROJECT_ROOT:?}"
: "${MV17B_OUTPUT_ROOT:?}"
: "${MV17B_PYTHON:?}"
CAMPAIGN="${MV17B_OUTPUT_ROOT}/campaign"
test -s "${MV17B_OUTPUT_ROOT}/analysis/summary.json"
exec > >(tee "${CAMPAIGN}/logs/post_${SLURM_JOB_ID}.out") 2>&1
cd "${MV17B_PROJECT_ROOT}"
export PATH="$(dirname "${MV17B_PYTHON}"):${PATH}"
export PYTHONPATH="${MV17B_PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
source "${MV17B_OUTPUT_ROOT}/SUBMISSION.env"
sacct -j "${MV17B_JOB_IDS}" -X -P \
  --format=JobID,JobIDRaw,JobName,State,ExitCode,ElapsedRaw,CPUTimeRAW,MaxRSS,NodeList \
  > "${MV17B_OUTPUT_ROOT}/analysis/mv17b_slurm_accounting.psv" || true
"${MV17B_PYTHON}" -m vgdsmc.mohammadzadeh_mv17b_fresh_cylinder_confirmation package \
  --output-root "${MV17B_OUTPUT_ROOT}" \
  --return-directory "${MV17B_PROJECT_ROOT}"
echo "MV17B_POST_PASS output=${MV17B_OUTPUT_ROOT}"

