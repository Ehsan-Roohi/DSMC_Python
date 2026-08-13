#!/usr/bin/env bash
#SBATCH --job-name=mv11-cyl-seed
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=36:00:00

set -euo pipefail
: "${MV11_CAMPAIGN_ROOT:?}"
: "${MV11_PAYLOAD_ROOT:?}"
: "${SLURM_ARRAY_TASK_ID:?}"

test -f "${MV11_CAMPAIGN_ROOT}/PREPARE_PASS"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MV11_CAMPAIGN_ROOT}/seed_table.tsv")
IFS=$'\t' read -r CASE_ID SEED <<< "${LINE}"
test -n "${CASE_ID}" && test -n "${SEED}"

CASE_DIR="${MV11_CAMPAIGN_ROOT}/cases/${CASE_ID}"
RESULT_DIR="${CASE_DIR}/results"
MOMENT_DIR="${RESULT_DIR}/moments"
FIELD_DIR="${RESULT_DIR}/fields"
SURFACE_DIR="${RESULT_DIR}/surface"
LOG_DIR="${CASE_DIR}/logs"
mkdir -p "${CASE_DIR}/input" "${MOMENT_DIR}" "${FIELD_DIR}" "${SURFACE_DIR}" "${LOG_DIR}"
exec > >(tee "${LOG_DIR}/slurm_${SLURM_JOB_ID}.out") 2>&1

LOCAL_BASE="${TMPDIR:-/tmp}"
LOCAL="${LOCAL_BASE}/mv11_ds2v_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if [[ "${LOCAL}" != "${LOCAL_BASE}"/mv11_ds2v_* ]]; then
  echo "unsafe local scratch path" >&2
  exit 90
fi
mkdir -p "${LOCAL}"

install -m 0755 "${MV11_CAMPAIGN_ROOT}/build/ds2v_bird_mv11" "${LOCAL}/ds2v"
install -m 0644 "${MV11_CAMPAIGN_ROOT}/input/DS2VD.DAT" "${LOCAL}/DS2VD.DAT"
install -m 0644 "${MV11_CAMPAIGN_ROOT}/input/interactive_input.txt" "${LOCAL}/interactive_input.txt"
install -m 0644 "${MV11_CAMPAIGN_ROOT}/input/HEAT-BENCH.TXT" "${LOCAL}/HEAT-BENCH.TXT"
ln -s HEAT-BENCH.TXT "${LOCAL}/HEAT-BENCH.txt"
printf '%s\n' "${SEED}" > "${LOCAL}/RANDOM_SEED.IN"
cp -a "${LOCAL}/DS2VD.DAT" "${LOCAL}/interactive_input.txt" "${LOCAL}/RANDOM_SEED.IN" "${CASE_DIR}/input/"

python3 - "${CASE_DIR}/CASE_METADATA.json" "${CASE_ID}" "${SEED}" "${SLURM_JOB_ID}" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "case_id": sys.argv[2], "seed": int(sys.argv[3]), "job_id": int(sys.argv[4]),
    "run_type": "fresh_IRUN_3", "restart_reused": False
}, indent=2) + "\n")
PY

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GFORTRAN_UNBUFFERED_ALL=1
MIN_NOUT="${MV11_MIN_NOUT:-20}"
MAX_NOUT="${MV11_MAX_NOUT:-60}"
TARGET_TUD="${MV11_TARGET_TUD:-30}"
UINF=2634.1
DIAMETER=0.3048
STATUS=RUNNING
LAST_NOUT=0
LAST_TIME=0
LAST_TUD=0
SOLVER_PID=""

write_status() {
  cat > "${RESULT_DIR}/RUN_STATUS.env" <<EOF
STATUS=${STATUS}
CASE_ID=${CASE_ID}
SEED=${SEED}
JOB_ID=${SLURM_JOB_ID}
LAST_NOUT=${LAST_NOUT}
LAST_TIME=${LAST_TIME}
LAST_TUD=${LAST_TUD}
MIN_NOUT=${MIN_NOUT}
MAX_NOUT=${MAX_NOUT}
TARGET_TUD=${TARGET_TUD}
EOF
}

sync_outputs() {
  local FILE BASE NOUT DEST COMPLETED
  local -a MOMENT_FILES=()
  mapfile -t MOMENT_FILES < <(find "${LOCAL}" -maxdepth 1 -type f -name 'MV11_MOMENTS_NOUT*.DAT' -printf '%f\n' | sort)
  ((${#MOMENT_FILES[@]} > 0)) || return 0
  COMPLETED=$(grep -c 'Output files written' "${LOCAL}/run.log" 2>/dev/null || true)
  # MV11 closes the moment file immediately before this marker. Never copy a
  # newly visible file while the Fortran WRITE is still in progress.
  (( COMPLETED >= ${#MOMENT_FILES[@]} )) || return 0
  for BASE in "${MOMENT_FILES[@]}"; do
    install -m 0644 "${LOCAL}/${BASE}" "${MOMENT_DIR}/${BASE}"
  done
  FILE=${MOMENT_FILES[-1]}
  NOUT=${FILE#MV11_MOMENTS_NOUT}; NOUT=${NOUT%.DAT}; NOUT=$((10#${NOUT}))
  if (( NOUT <= LAST_NOUT )); then return 0; fi
  LAST_NOUT=${NOUT}
  LAST_TIME=$(sed -n '2s/.* TIME=\([^ ]*\).*/\1/p' "${LOCAL}/${FILE}")
  LAST_TIME=${LAST_TIME:-0}
  LAST_TUD=$(awk -v T="${LAST_TIME}" -v U="${UINF}" -v D="${DIAMETER}" 'BEGIN{printf "%.12g",T*U/D}')
  DEST="NOUT$(printf '%04d' "${LAST_NOUT}")"
  [[ -s "${LOCAL}/DS2FF.DAT" ]] && install -m 0644 "${LOCAL}/DS2FF.DAT" "${FIELD_DIR}/DS2FF_${DEST}.DAT"
  mkdir -p "${SURFACE_DIR}/${DEST}"
  for BASE in "DS2SU - Copy.DAT" "DS2SUJUMP.DAT" "DS2FF_ON_SURFACE_CURVE.DAT" "Lift&Drag&moment.dat" "HEAT FLUX ERROR.dat"; do
    [[ -s "${LOCAL}/${BASE}" ]] && install -m 0644 "${LOCAL}/${BASE}" "${SURFACE_DIR}/${DEST}/${BASE}"
  done
  write_status
  echo "MV11_SNAPSHOT_PASS case=${CASE_ID} nout=${LAST_NOUT} time=${LAST_TIME} tud=${LAST_TUD}"
}

finalize() {
  set +e
  if [[ -n "${SOLVER_PID}" ]] && kill -0 "${SOLVER_PID}" 2>/dev/null; then
    kill -TERM "${SOLVER_PID}" 2>/dev/null || true
    sleep 3
    kill -KILL "${SOLVER_PID}" 2>/dev/null || true
  fi
  [[ -n "${SOLVER_PID}" ]] && wait "${SOLVER_PID}" 2>/dev/null || true
  sync_outputs
  for FILE in run.log RNG_SEED_USED.txt RNG_STATE.DAT DS2VD.TXT DIAG.TXT "FINAL TIME.dat"; do
    [[ -s "${LOCAL}/${FILE}" ]] && cp -a "${LOCAL}/${FILE}" "${RESULT_DIR}/"
  done
  write_status
  find "${CASE_DIR}" -type f ! -name 'CASE_SHA256SUMS.txt' -print0 | sort -z | xargs -0 sha256sum > "${CASE_DIR}/CASE_SHA256SUMS.txt" 2>/dev/null || true
}
on_signal() {
  STATUS=SIGNALLED
  exit 143
}
trap finalize EXIT
trap on_signal TERM INT

cd "${LOCAL}"
./ds2v < interactive_input.txt > run.log 2>&1 &
SOLVER_PID=$!
write_status
DEADLINE=$((SECONDS + 34 * 3600))

while kill -0 "${SOLVER_PID}" 2>/dev/null; do
  sync_outputs
  if (( LAST_NOUT >= MIN_NOUT )) && awk -v X="${LAST_TUD}" -v T="${TARGET_TUD}" 'BEGIN{exit !(X>=T)}'; then
    STATUS=TARGET_REACHED
    echo "MV11_TARGET_REACHED case=${CASE_ID} nout=${LAST_NOUT} tud=${LAST_TUD}"
    exit 0
  fi
  if (( LAST_NOUT >= MAX_NOUT )); then STATUS=MAX_NOUT_REACHED; exit 0; fi
  if grep -qiE 'Fortran runtime error|Error termination|Segmentation fault|Floating point exception|FATAL:' run.log 2>/dev/null; then
    STATUS=FATAL
    tail -n 200 run.log || true
    exit 20
  fi
  if (( SECONDS >= DEADLINE )); then STATUS=PARTIAL_WALLTIME; exit 0; fi
  sleep 10
done

wait "${SOLVER_PID}" || SOLVER_RC=$?
SOLVER_RC=${SOLVER_RC:-0}
sync_outputs
if (( LAST_NOUT >= MIN_NOUT )); then
  STATUS=SOLVER_EXITED_WITH_USABLE_BLOCKS
  exit 0
fi
STATUS=SOLVER_EXITED_EARLY
exit "${SOLVER_RC:-21}"
