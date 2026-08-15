#!/usr/bin/env bash
#SBATCH --job-name=moh-mv17a-post
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
: "${MV17A_PROJECT_ROOT:?}"
: "${MV17A_OUTPUT_ROOT:?}"
: "${MV17A_PYTHON:?}"
: "${MV17A_AUDIT_JOB_ID:?}"
test -x "${MV17A_PYTHON}"
test -s "${MV17A_OUTPUT_ROOT}/summary.json"

exec > >(tee "${MV17A_PROJECT_ROOT}/logs/moh_mv17a_post_${SLURM_JOB_ID}.out") 2>&1
cd "${MV17A_PROJECT_ROOT}"
export PATH="$(dirname "${MV17A_PYTHON}"):${PATH}"
export PYTHONPATH="${MV17A_PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
sacct -j "${MV17A_AUDIT_JOB_ID},${SLURM_JOB_ID}" -X -P \
  --format=JobID,JobIDRaw,JobName,State,ExitCode,ElapsedRaw,CPUTimeRAW,MaxRSS,NodeList \
  > "${MV17A_OUTPUT_ROOT}/mv17a_slurm_accounting.psv" || true
"${MV17A_PYTHON}" -m vgdsmc.mohammadzadeh_mv17a_cylinder_native_crossfit package \
  --output-root "${MV17A_OUTPUT_ROOT}" \
  --return-directory "${MV17A_PROJECT_ROOT}"
echo "MV17A_POST_PASS output=${MV17A_OUTPUT_ROOT}"
