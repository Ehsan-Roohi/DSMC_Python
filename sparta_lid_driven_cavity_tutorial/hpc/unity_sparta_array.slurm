#!/usr/bin/env bash
#SBATCH --job-name=sparta-cavity
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/sparta-cavity-%A_%a.out
#SBATCH --error=logs/sparta-cavity-%A_%a.err

set -euo pipefail

ROOT_DIR="${SPARTA_TUTORIAL_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${ROOT_DIR}" || ! -d "${ROOT_DIR}" ]]; then
  echo "SPARTA_TUTORIAL_ROOT/SLURM_SUBMIT_DIR does not identify the tutorial directory." >&2
  exit 2
fi
ROOT_DIR="$(cd "${ROOT_DIR}" && pwd)"
SPARTA_BIN="${SPARTA_BIN:-${ROOT_DIR}/third_party/sparta/src/spa_mpi}"
OPENMPI_MODULE="${UNITY_OPENMPI_MODULE:-openmpi/5.0.3}"
MPI_LAUNCHER="${UNITY_MPI_LAUNCHER:-mpirun}"
SEEDS=(20260803 20260819 20260831)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#SEEDS[@]} )); then
  echo "SLURM_ARRAY_TASK_ID must select one of ${#SEEDS[@]} seeds." >&2
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
RUN_DIR="${ROOT_DIR}/runs/production_kn01_seed_${SEED}"
mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/runs"

if [[ -e "${RUN_DIR}" ]]; then
  if [[ -s "${RUN_DIR}/grid.final.00026000" ]]; then
    echo "Refusing to overwrite completed run: ${RUN_DIR}" >&2
    exit 4
  fi
  PRESERVED="${RUN_DIR}.incomplete_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID}"
  mv -- "${RUN_DIR}" "${PRESERVED}"
  echo "Preserved incomplete run at: ${PRESERVED}"
fi

python3 "${ROOT_DIR}/scripts/generate_case.py" \
  --level production --kn 0.1 --seed "${SEED}" --output "${RUN_DIR}"

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
set +e
"${MPI_LAUNCHER}" -n 2 "${SPARTA_BIN}" -help > mpi_preflight.out 2> mpi_preflight.err
PREFLIGHT_STATUS=$?
set -e
if ! grep -q "^Usage: sparta" mpi_preflight.out; then
  echo "MPI/SPARTA preflight failed (status ${PREFLIGHT_STATUS})." >&2
  tail -n 80 mpi_preflight.err >&2 || true
  exit 7
fi

# Standard input avoids Open MPI/PRRTE treating SPARTA's -log option as its own.
"${MPI_LAUNCHER}" -n "${SLURM_NTASKS}" "${SPARTA_BIN}" < in.cavity
if [[ ! -s log.sparta ]]; then
  echo "SPARTA ended without log.sparta." >&2
  exit 8
fi
mv -- log.sparta log.cavity

if [[ ! -s grid.final.00026000 ]]; then
  echo "Expected final dump grid.final.00026000 was not created." >&2
  exit 5
fi

python3 "${ROOT_DIR}/scripts/postprocess.py" "${RUN_DIR}"
printf 'finished_utc=%s\nstatus=complete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> unity_run_metadata.txt
echo "COMPLETE: ${RUN_DIR}"

