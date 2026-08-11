#!/usr/bin/env bash
#SBATCH --job-name=moh_mv6_pub
#SBATCH --account=pi_roohie_umass_edu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${MV6_REPO_ROOT:?exported by installer}"
cd "${REPO_ROOT}"

if [[ -f LAST_MOHAMMADZADEH_MV6_POSTFIX_JOB.env ]]; then
    source LAST_MOHAMMADZADEH_MV6_POSTFIX_JOB.env
elif [[ -f LAST_MOHAMMADZADEH_MV5_REPAIR_MV6_JOB.env ]]; then
    source LAST_MOHAMMADZADEH_MV5_REPAIR_MV6_JOB.env
else
    echo "MV6 environment file is absent" >&2
    exit 2
fi

: "${MV6_OUTPUT_ROOT:?MV6_OUTPUT_ROOT is absent}"
export MV6_OUTPUT_ROOT
export MV6_ARCHIVE_DIR="${HOME}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mv6_publication_matplotlib_${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export PYTHONUNBUFFERED=1

mkdir -p "${MPLCONFIGDIR}"

if [[ -n "${MV6_VENV_DIR:-}" && -x "${MV6_VENV_DIR}/bin/python" ]]; then
    source "${MV6_VENV_DIR}/bin/activate"
fi

python -m py_compile scripts/make_mv6_publication_suite.py
python scripts/make_mv6_publication_suite.py
