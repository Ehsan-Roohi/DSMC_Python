#!/usr/bin/env bash
#SBATCH --job-name=moh-mv16b-post
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
: "${MV16B_PROJECT_ROOT:?}"
: "${MV16B_OUTPUT_ROOT:?}"
: "${MV16B_PYTHON:?}"
: "${MV16B_AUDIT_JOB_ID:?}"
test -x "${MV16B_PYTHON}"
test -s "${MV16B_OUTPUT_ROOT}/summary.json"

exec > >(tee "${MV16B_PROJECT_ROOT}/logs/moh_mv16b_post_${SLURM_JOB_ID}.out") 2>&1
cd "${MV16B_PROJECT_ROOT}"
export PATH="$(dirname "${MV16B_PYTHON}"):${PATH}"
export PYTHONPATH="${MV16B_PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
sacct -j "${MV16B_AUDIT_JOB_ID},${SLURM_JOB_ID}" -X -P \
  --format=JobID,JobIDRaw,JobName,State,ExitCode,ElapsedRaw,CPUTimeRAW,MaxRSS,NodeList \
  > "${MV16B_OUTPUT_ROOT}/mv16b_slurm_accounting.psv" || true
"${MV16B_PYTHON}" -m vgdsmc.mohammadzadeh_mv16b_jcp_evidence_audit package \
  --output-root "${MV16B_OUTPUT_ROOT}" \
  --return-directory "${MV16B_PROJECT_ROOT}"
echo "MV16B_POST_PASS output=${MV16B_OUTPUT_ROOT}"

