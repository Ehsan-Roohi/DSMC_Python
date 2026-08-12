#!/usr/bin/env bash
#SBATCH --job-name=moh_mv7_pub
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

REPO_ROOT="${MV7_REPO_ROOT:?exported by installer}"
cd "${REPO_ROOT}"
source LAST_MOHAMMADZADEH_MV7_JCP_JOB.env

: "${MV7_OUTPUT_ROOT:?MV7_OUTPUT_ROOT is absent}"
: "${MV7_MV6_ROOT:?MV7_MV6_ROOT is absent}"
: "${MV7_M3_ROOT:?MV7_M3_ROOT is absent}"
: "${MV7_MV3_ROOT:?MV7_MV3_ROOT is absent}"
: "${MV7_REFERENCE_ROOT:?MV7_REFERENCE_ROOT is absent}"

export MV7_REPO_ROOT="${REPO_ROOT}"
export MV7_ARCHIVE_DIR="${HOME}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mv7_publication_matplotlib_${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export PYTHONUNBUFFERED=1
mkdir -p "${MPLCONFIGDIR}"

if [[ -n "${MV7_VENV_DIR:-}" && -x "${MV7_VENV_DIR}/bin/python" ]]; then
    source "${MV7_VENV_DIR}/bin/activate"
fi

python -m py_compile scripts/make_mv7_jcp_publication_suite.py
python -m pytest -q tests/test_make_mv7_jcp_publication_suite.py
python scripts/make_mv7_jcp_publication_suite.py
