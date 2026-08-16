#!/usr/bin/env bash
#SBATCH --job-name=sp-shock-prod
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-8%3
#SBATCH --output=logs/sp-shock-prod-%A_%a.out
#SBATCH --error=logs/sp-shock-prod-%A_%a.err

set -euo pipefail

ROOT_DIR="${SPARTA_CASE_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "${ROOT_DIR}" && -d "${ROOT_DIR}" ]] || {
  echo "SPARTA_CASE_ROOT/SLURM_SUBMIT_DIR does not identify the case directory." >&2
  exit 2
}
ROOT_DIR="$(cd "${ROOT_DIR}" && pwd)"
SPARTA_BIN="${SPARTA_BIN:-${ROOT_DIR}/third_party/sparta/src/spa_mpi}"
OPENMPI_MODULE="${UNITY_OPENMPI_MODULE:-openmpi/5.0.3}"
MPI_LAUNCHER="${UNITY_MPI_LAUNCHER:-mpirun}"
MACHS=(2.5 2.5 2.5 3.0 3.0 3.0 5.0 5.0 5.0)
SEEDS=(20260803 20260819 20260831 20260803 20260819 20260831 20260803 20260819 20260831)
MACH_LABELS=(M25 M25 M25 M30 M30 M30 M50 M50 M50)
FINAL_DUMP_NAME="profile.final.00160000"

TASK_ID="${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array task}"
(( TASK_ID >= 0 && TASK_ID < ${#MACHS[@]} )) || { echo "Invalid task index: ${TASK_ID}" >&2; exit 2; }
module purge
module load "${OPENMPI_MODULE}"
module list
for command_name in "${MPI_LAUNCHER}" python3; do
  command -v "${command_name}" >/dev/null 2>&1 || { echo "Missing ${command_name}" >&2; exit 2; }
done
[[ -x "${SPARTA_BIN}" ]] || { echo "Missing MPI SPARTA executable: ${SPARTA_BIN}" >&2; exit 3; }

MACH="${MACHS[TASK_ID]}"
MACH_LABEL="${MACH_LABELS[TASK_ID]}"
SEED="${SEEDS[TASK_ID]}"
ARRAY_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
RUN_NAME="shock_${ARRAY_ID}_${MACH_LABEL}_seed_${SEED}"
RUN_DIR="${ROOT_DIR}/runs/${RUN_NAME}"
mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/runs"
[[ ! -e "${RUN_DIR}" ]] || { echo "Refusing to overwrite ${RUN_DIR}" >&2; exit 4; }

python3 "${ROOT_DIR}/scripts/generate_case.py" \
  --level production --mach "${MACH}" --seed "${SEED}" --output "${RUN_DIR}" \
  > "${RUN_DIR}.generator.json"
mv "${RUN_DIR}.generator.json" "${RUN_DIR}/generator_output.json"

python3 - "${RUN_DIR}/case_metadata.json" "${MACH}" "${SEED}" <<'PY'
import json
import math
import sys
metadata = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "level": "production",
    "nx": 600,
    "ny": 1,
    "target_upstream_particles_per_cell": 64,
    "warmup_steps": 40000,
    "sample_steps": 160000,
    "sample_stride": 10,
    "dump_frequency_steps": 40000,
    "restart_frequency_steps": 40000,
    "seed": int(sys.argv[3]),
}
bad = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
if not math.isclose(float(metadata["mach_1"]), float(sys.argv[2]), rel_tol=0.0, abs_tol=1e-12):
    bad["mach_1"] = (metadata["mach_1"], float(sys.argv[2]))
if bad:
    raise SystemExit(f"Production case contract failed: {bad}")
print("SPARTA_NORMAL_SHOCK_CASE_CONTRACT_PASS")
PY

{
  printf 'array_job_id=%s\n' "${ARRAY_ID}"
  printf 'array_task_id=%s\n' "${TASK_ID}"
  printf 'mach=%s\n' "${MACH}"
  printf 'seed=%s\n' "${SEED}"
  printf 'configuration=steady_normal_shock_nx600_ppc64_warmup40000_average160000_stride10\n'
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
grep -q '^Usage: sparta' mpi_preflight.out || {
  echo "MPI/SPARTA preflight failed (status ${PREFLIGHT_STATUS})." >&2
  tail -n 80 mpi_preflight.err >&2 || true
  exit 7
}
printf 'mpi_preflight=pass\n' >> unity_run_metadata.txt

"${MPI_LAUNCHER}" -n "${SLURM_NTASKS}" "${SPARTA_BIN}" < in.shock
[[ -s log.sparta ]] || { echo "SPARTA ended without log.sparta" >&2; exit 8; }
mv -- log.sparta log.shock
[[ -s "${FINAL_DUMP_NAME}" ]] || { echo "Missing final dump ${FINAL_DUMP_NAME}" >&2; exit 5; }
python3 "${ROOT_DIR}/scripts/postprocess.py" single "${RUN_DIR}" > postprocess_output.json
PHYSICS_GATE="$(python3 -c 'import json; print(str(json.load(open("validation_metrics.json"))["physics_gate_pass"]).lower())')"
{
  printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'final_dump=%s\n' "${FINAL_DUMP_NAME}"
  printf 'physics_gate_pass=%s\n' "${PHYSICS_GATE}"
  printf 'status=complete\n'
} >> unity_run_metadata.txt
echo "COMPLETE: ${RUN_DIR} physics_gate_pass=${PHYSICS_GATE}"

