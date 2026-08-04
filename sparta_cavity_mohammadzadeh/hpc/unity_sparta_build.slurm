#!/usr/bin/env bash
#SBATCH --job-name=sparta-build
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sparta-build-%j.out
#SBATCH --error=logs/sparta-build-%j.err

set -euo pipefail

ROOT_DIR="${SPARTA_CASE_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${ROOT_DIR}" || ! -d "${ROOT_DIR}" ]]; then
  echo "SPARTA_CASE_ROOT/SLURM_SUBMIT_DIR does not identify the case directory." >&2
  exit 2
fi
ROOT_DIR="$(cd "${ROOT_DIR}" && pwd)"
SPARTA_BIN="${ROOT_DIR}/third_party/sparta/src/spa_mpi"
OPENMPI_MODULE="${UNITY_OPENMPI_MODULE:-openmpi/5.0.3}"
BUILD_JOBS="${SLURM_CPUS_PER_TASK:-8}"

module purge
module load "${OPENMPI_MODULE}"
module list

for COMMAND_NAME in git make g++ mpicxx srun; do
  if ! command -v "${COMMAND_NAME}" >/dev/null 2>&1; then
    echo "Required build command is unavailable after loading ${OPENMPI_MODULE}: ${COMMAND_NAME}" >&2
    exit 2
  fi
done

if [[ -x "${SPARTA_BIN}" ]]; then
  echo "Reusing existing MPI-enabled SPARTA executable: ${SPARTA_BIN}"
elif [[ -d "${ROOT_DIR}/third_party/sparta/src" ]]; then
  make -C "${ROOT_DIR}/third_party/sparta/src" -j"${BUILD_JOBS}" mpi
else
  BUILD_JOBS="${BUILD_JOBS}" bash "${ROOT_DIR}/scripts/install_sparta_linux.sh" mpi
fi

if [[ ! -x "${SPARTA_BIN}" ]]; then
  echo "SPARTA build did not create ${SPARTA_BIN}" >&2
  exit 3
fi

"${SPARTA_BIN}" -help > "${ROOT_DIR}/logs/sparta-build-${SLURM_JOB_ID}-help.txt"
echo "READY: ${SPARTA_BIN}"
