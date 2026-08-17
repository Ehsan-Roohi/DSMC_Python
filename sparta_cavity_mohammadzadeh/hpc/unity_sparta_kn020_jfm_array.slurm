#!/usr/bin/env bash
#SBATCH --job-name=sp-k020-jfm
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/sp-k020-jfm-%A_%a.out
#SBATCH --error=logs/sp-k020-jfm-%A_%a.err

set -euo pipefail

ROOT_DIR="${SPARTA_CASE_ROOT:-${SLURM_SUBMIT_DIR:-}}"
RESULTS_BASE="${SPARTA_RESULTS_BASE:?SPARTA_RESULTS_BASE was not exported}"
if [[ -z "${ROOT_DIR}" || ! -d "${ROOT_DIR}" ]]; then
  echo "SPARTA_CASE_ROOT/SLURM_SUBMIT_DIR does not identify the code directory." >&2
  exit 2
fi
ROOT_DIR="$(cd "${ROOT_DIR}" && pwd)"
SPARTA_BIN="${SPARTA_BIN:-${ROOT_DIR}/third_party/sparta/src/spa_mpi}"
OPENMPI_MODULE="${UNITY_OPENMPI_MODULE:-openmpi/5.0.3}"
MPI_LAUNCHER="${UNITY_MPI_LAUNCHER:-mpirun}"
SEEDS=(104729 130363)
FINAL_DUMP_NAME="grid.final.00200000"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]] || \
   (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#SEEDS[@]} )); then
  echo "SLURM_ARRAY_TASK_ID must select one of ${#SEEDS[@]} seeds." >&2
  exit 2
fi

module purge
module load "${OPENMPI_MODULE}"
module list
export OMP_NUM_THREADS=1

for command_name in "${MPI_LAUNCHER}" python3; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Required runtime command is unavailable: ${command_name}" >&2
    exit 2
  fi
done
if [[ ! -x "${SPARTA_BIN}" ]]; then
  echo "Missing MPI-enabled SPARTA executable: ${SPARTA_BIN}" >&2
  exit 3
fi

SEED="${SEEDS[SLURM_ARRAY_TASK_ID]}"
ARRAY_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
ARRAY_ROOT="${RESULTS_BASE}/array_${ARRAY_ID}"
RUN_DIR="${ARRAY_ROOT}/seed_${SEED}"
mkdir -p "${ARRAY_ROOT}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to overwrite an existing ensemble member: ${RUN_DIR}" >&2
  exit 4
fi

python3 "${ROOT_DIR}/scripts/generate_jfm_kn020_case.py" \
  --seed "${SEED}" --output "${RUN_DIR}"

python3 - "${RUN_DIR}/case_metadata.json" <<'PY'
import json
import sys

metadata = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "kn": 0.2,
    "kn_convention": "gu_lambda_over_L",
    "nx": 160,
    "ny": 160,
    "particles_per_cell": 256,
    "warmup_steps": 40000,
    "sample_steps": 200000,
    "sample_stride": 10,
    "accumulated_samples_per_cell": 20000,
    "checkpoint_frequency_steps": 40000,
    "argon_mass_kg": 6.6335e-26,
}
bad = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
if bad:
    raise SystemExit(f"Kn=0.20 publication contract failed: {bad}")
print("KN020_JFM_CASE_CONTRACT_PASS")
PY

{
  printf 'array_job_id=%s\n' "${ARRAY_ID}"
  printf 'array_task_id=%s\n' "${SLURM_ARRAY_TASK_ID}"
  printf 'seed=%s\n' "${SEED}"
  printf 'configuration=kn0.20_gu_N160_ppc256_warmup40000_samples20000_stride10\n'
  printf 'hostname=%s\n' "$(hostname)"
  printf 'slurm_ntasks=%s\n' "${SLURM_NTASKS}"
  printf 'openmpi_module=%s\n' "${OPENMPI_MODULE}"
  printf 'mpi_launcher=%s\n' "${MPI_LAUNCHER}"
  printf 'sparta_binary=%s\n' "${SPARTA_BIN}"
  printf 'code_commit=%s\n' "$(git -C "${ROOT_DIR}/.." rev-parse HEAD)"
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
printf 'mpi_preflight_status=%s\nmpi_preflight=pass\n' "${MPI_PREFLIGHT_STATUS}" \
  >> unity_run_metadata.txt

set +e
"${MPI_LAUNCHER}" -n "${SLURM_NTASKS}" "${SPARTA_BIN}" < in.cavity
RUN_STATUS=$?
set -e
if (( RUN_STATUS != 0 )); then
  printf 'finished_utc=%s\nstatus=failed\nsparta_exit=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${RUN_STATUS}" >> unity_run_metadata.txt
  exit "${RUN_STATUS}"
fi
if [[ ! -s log.sparta ]]; then
  echo "SPARTA ended without its default log.sparta file." >&2
  exit 8
fi
mv -- log.sparta log.cavity

if [[ ! -s "${RUN_DIR}/${FINAL_DUMP_NAME}" ]] || \
   ! grep -Fq 'f_fieldavg[7]' "${RUN_DIR}/${FINAL_DUMP_NAME}"; then
  echo "SPARTA ended without the required seven-field final dump." >&2
  exit 9
fi

{
  printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'final_dump=%s\n' "${FINAL_DUMP_NAME}"
  printf 'status=complete\n'
} >> unity_run_metadata.txt

echo "COMPLETE: ${RUN_DIR}"
