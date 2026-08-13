#!/usr/bin/env bash
#SBATCH --job-name=moh_mv10_post
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd "${MV10_REPO_ROOT:?}"
source "${MV10_VENV_DIR:?}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mv10_matplotlib_${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
mkdir -p "${MPLCONFIGDIR}"

if ! sacct -j "${MV10_ASSEMBLE_JOB_ID:?},${MV10_MODEL_JOB_ID:?}" -X -P \
  --format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,CPUTimeRAW,MaxRSS,NodeList \
  > "${MV10_OUTPUT_ROOT:?}/slurm_accounting.psv"; then
  printf 'JobIDRaw|JobName|State|ExitCode|ElapsedRaw|CPUTimeRAW|MaxRSS|NodeList|\n' \
    > "${MV10_OUTPUT_ROOT:?}/slurm_accounting.psv"
fi

python -m vgdsmc.mohammadzadeh_mv10_qy_multiscale \
  --mode post \
  --output-root "${MV10_OUTPUT_ROOT:?}" \
  --return-directory "${MV10_REPO_ROOT:?}"
