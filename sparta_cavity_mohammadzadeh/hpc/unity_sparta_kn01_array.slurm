#!/usr/bin/env bash
#SBATCH --job-name=sparta-kn01
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/sparta-kn01-%A_%a.out
#SBATCH --error=logs/sparta-kn01-%A_%a.err

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPARTA_BIN="${SPARTA_BIN:-${ROOT_DIR}/third_party/sparta/src/spa_mpi}"
OPENMPI_MODULE="${UNITY_OPENMPI_MODULE:-openmpi/5.0.3}"
SEEDS=(20260803 20260819 20260831)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#SEEDS[@]} )); then
  echo "SLURM_ARRAY_TASK_ID must select one of ${#SEEDS[@]} production seeds." >&2
  exit 2
fi

module purge
module load "${OPENMPI_MODULE}"
module list

if [[ ! -x "${SPARTA_BIN}" ]]; then
  echo "Missing MPI-enabled SPARTA executable: ${SPARTA_BIN}" >&2
  exit 3
fi

SEED="${SEEDS[SLURM_ARRAY_TASK_ID]}"
RUN_NAME="production_kn01_seed_${SEED}"
RUN_DIR="${ROOT_DIR}/runs/${RUN_NAME}"

mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/runs"
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to overwrite an existing production run: ${RUN_DIR}" >&2
  echo "Move or rename that directory before resubmitting this seed." >&2
  exit 4
fi

python3 "${ROOT_DIR}/scripts/generate_case.py" \
  --level production \
  --kn 0.1 \
  --seed "${SEED}" \
  --output "${RUN_DIR}"

{
  printf 'array_job_id=%s\n' "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
  printf 'array_task_id=%s\n' "${SLURM_ARRAY_TASK_ID}"
  printf 'seed=%s\n' "${SEED}"
  printf 'hostname=%s\n' "$(hostname)"
  printf 'slurm_ntasks=%s\n' "${SLURM_NTASKS}"
  printf 'openmpi_module=%s\n' "${OPENMPI_MODULE}"
  printf 'sparta_binary=%s\n' "${SPARTA_BIN}"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${RUN_DIR}/unity_run_metadata.txt"

cd "${RUN_DIR}"
srun "${SPARTA_BIN}" -in in.cavity -log log.cavity

FINAL_DUMP="${RUN_DIR}/grid.final.00026000"
if [[ ! -s "${FINAL_DUMP}" ]]; then
  echo "SPARTA ended without the expected final grid dump: ${FINAL_DUMP}" >&2
  exit 5
fi

{
  printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'final_dump=%s\n' "$(basename "${FINAL_DUMP}")"
  printf 'status=complete\n'
} >> "${RUN_DIR}/unity_run_metadata.txt"

echo "COMPLETE: ${RUN_DIR}"
