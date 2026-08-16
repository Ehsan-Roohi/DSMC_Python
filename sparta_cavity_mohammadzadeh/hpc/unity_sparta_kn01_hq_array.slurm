#!/usr/bin/env bash
#SBATCH --job-name=sparta-kn01-hq
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/sparta-kn01-hq-%A_%a.out
#SBATCH --error=logs/sparta-kn01-hq-%A_%a.err

set -euo pipefail

ROOT_DIR="${SPARTA_CASE_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${ROOT_DIR}" || ! -d "${ROOT_DIR}" ]]; then
  echo "SPARTA_CASE_ROOT/SLURM_SUBMIT_DIR does not identify the case directory." >&2
  exit 2
fi
ROOT_DIR="$(cd "${ROOT_DIR}" && pwd)"
SPARTA_BIN="${SPARTA_BIN:-${ROOT_DIR}/third_party/sparta/src/spa_mpi}"
OPENMPI_MODULE="${UNITY_OPENMPI_MODULE:-openmpi/5.0.3}"
MPI_LAUNCHER="${UNITY_MPI_LAUNCHER:-mpirun}"
SEEDS=(20260803 20260819 20260831)
FINAL_DUMP_NAME="grid.final.00160000"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#SEEDS[@]} )); then
  echo "SLURM_ARRAY_TASK_ID must select one of ${#SEEDS[@]} HQ seeds." >&2
  exit 2
fi

module purge
module load "${OPENMPI_MODULE}"
module list

for COMMAND_NAME in "${MPI_LAUNCHER}" python3; do
  if ! command -v "${COMMAND_NAME}" >/dev/null 2>&1; then
    echo "Required runtime command is unavailable after loading ${OPENMPI_MODULE}: ${COMMAND_NAME}" >&2
    exit 2
  fi
done
if [[ ! -x "${SPARTA_BIN}" ]]; then
  echo "Missing MPI-enabled SPARTA executable: ${SPARTA_BIN}" >&2
  exit 3
fi

SEED="${SEEDS[SLURM_ARRAY_TASK_ID]}"
ARRAY_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
RUN_NAME="hq_${ARRAY_ID}_kn01_seed_${SEED}"
RUN_DIR="${ROOT_DIR}/runs/${RUN_NAME}"

mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/runs"
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to overwrite an existing campaign member: ${RUN_DIR}" >&2
  exit 4
fi

python3 "${ROOT_DIR}/scripts/generate_case.py" \
  --level hq \
  --kn 0.1 \
  --seed "${SEED}" \
  --output "${RUN_DIR}"

python3 - "${RUN_DIR}/case_metadata.json" <<'PY'
import json
import sys

metadata = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "kn": 0.1,
    "nx": 200,
    "ny": 200,
    "particles_per_cell": 128,
    "warmup_steps": 40000,
    "sample_steps": 160000,
    "sample_stride": 10,
    "temperature_observable": "thermal/grid COM-subtracted translational temperature",
}
bad = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
if bad:
    raise SystemExit(f"HQ case contract failed: {bad}")
print("HQ_CASE_CONTRACT_PASS")
PY

{
  printf 'array_job_id=%s\n' "${ARRAY_ID}"
  printf 'array_task_id=%s\n' "${SLURM_ARRAY_TASK_ID}"
  printf 'seed=%s\n' "${SEED}"
  printf 'configuration=grid200x200_ppc128_warmup40000_average160000_stride10\n'
  printf 'hostname=%s\n' "$(hostname)"
  printf 'slurm_ntasks=%s\n' "${SLURM_NTASKS}"
  printf 'openmpi_module=%s\n' "${OPENMPI_MODULE}"
  printf 'mpi_launcher=%s\n' "${MPI_LAUNCHER}"
  printf 'sparta_binary=%s\n' "${SPARTA_BIN}"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${RUN_DIR}/unity_run_metadata.txt"

cd "${RUN_DIR}"
set +e
"${MPI_LAUNCHER}" -n 2 "${SPARTA_BIN}" -help > mpi_preflight.out 2> mpi_preflight.err
MPI_PREFLIGHT_STATUS=$?
set -e
if ! grep -q "^Usage: sparta" mpi_preflight.out; then
  echo "MPI/SPARTA preflight failed (launcher status ${MPI_PREFLIGHT_STATUS})." >&2
  tail -n 80 mpi_preflight.err >&2 || true
  exit 7
fi
printf 'mpi_preflight_status=%s\n' "${MPI_PREFLIGHT_STATUS}" >> unity_run_metadata.txt
printf 'mpi_preflight=pass\n' >> unity_run_metadata.txt

"${MPI_LAUNCHER}" -n "${SLURM_NTASKS}" "${SPARTA_BIN}" < in.cavity
if [[ ! -s log.sparta ]]; then
  echo "SPARTA ended without its default log.sparta file." >&2
  exit 8
fi
mv -- log.sparta log.cavity

FINAL_DUMP="${RUN_DIR}/${FINAL_DUMP_NAME}"
if [[ ! -s "${FINAL_DUMP}" ]]; then
  echo "SPARTA ended without the expected HQ grid dump: ${FINAL_DUMP}" >&2
  exit 5
fi

{
  printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'final_dump=%s\n' "${FINAL_DUMP_NAME}"
  printf 'status=complete\n'
} >> unity_run_metadata.txt

echo "COMPLETE: ${RUN_DIR}"
