#!/usr/bin/env bash
#SBATCH --job-name=sparta-cavity
#SBATCH --partition=cpu
#SBATCH --output=logs/sparta-cavity-%j.out
#SBATCH --error=logs/sparta-cavity-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT_DIR}/logs"

module purge
module load gcc
module load openmpi

export MPI_RANKS="${SLURM_NTASKS}"
export SPARTA_SKIP_POST=1
bash "${ROOT_DIR}/scripts/run_case.sh" production mpi

python3 "${ROOT_DIR}/scripts/postprocess.py" "${ROOT_DIR}/runs/production_kn01"
