#!/usr/bin/env bash
#SBATCH --job-name=moh-mv17b-run
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=12:00:00

set -euo pipefail
: "${MV17B_OUTPUT_ROOT:?}"
: "${SLURM_ARRAY_TASK_ID:?}"
CAMPAIGN="${MV17B_OUTPUT_ROOT}/campaign"
test -f "${CAMPAIGN}/PREPARE_PASS"
test -s "${MV17B_OUTPUT_ROOT}/frozen/FROZEN_MODEL_PASS"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MV17B_OUTPUT_ROOT}/seed_table.tsv")
IFS=$'\t' read -r CASE_ID PAIR_ID ROLE SEED <<< "${LINE}"
test -n "${CASE_ID}" && test -n "${PAIR_ID}" && test -n "${ROLE}" && test -n "${SEED}"

CASE_DIR="${CAMPAIGN}/cases/${CASE_ID}"
RESULT_DIR="${CASE_DIR}/results"
MOMENT_DIR="${RESULT_DIR}/moments"
LOG_DIR="${CASE_DIR}/logs"
mkdir -p "${CASE_DIR}/input" "${MOMENT_DIR}" "${LOG_DIR}"
exec > >(tee "${LOG_DIR}/slurm_${SLURM_JOB_ID}.out") 2>&1

LOCAL_BASE="${TMPDIR:-/tmp}"
LOCAL="${LOCAL_BASE}/mv17b_ds2v_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if [[ "${LOCAL}" != "${LOCAL_BASE}"/mv17b_ds2v_* ]]; then
  echo "MV17B unsafe local scratch path" >&2
  exit 90
fi
mkdir -p "${LOCAL}"
install -m 0755 "${CAMPAIGN}/build/ds2v_bird_mv17b" "${LOCAL}/ds2v"
install -m 0644 "${CAMPAIGN}/input/DS2VD.DAT" "${LOCAL}/DS2VD.DAT"
install -m 0644 "${CAMPAIGN}/input/interactive_input.txt" "${LOCAL}/interactive_input.txt"
install -m 0644 "${CAMPAIGN}/input/HEAT-BENCH.TXT" "${LOCAL}/HEAT-BENCH.TXT"
ln -s HEAT-BENCH.TXT "${LOCAL}/HEAT-BENCH.txt"
printf '%s\n' "${SEED}" > "${LOCAL}/RANDOM_SEED.IN"
cp -a "${LOCAL}/DS2VD.DAT" "${LOCAL}/interactive_input.txt" "${LOCAL}/RANDOM_SEED.IN" "${CASE_DIR}/input/"

python3 - "${CASE_DIR}/CASE_METADATA.json" "${CASE_ID}" "${PAIR_ID}" "${ROLE}" "${SEED}" "${SLURM_JOB_ID}" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "case_id": sys.argv[2], "pair_id": sys.argv[3], "role": sys.argv[4],
    "seed": int(sys.argv[5]), "job_id": int(sys.argv[6]),
    "run_type": "fresh_IRUN_3", "restart_reused": False,
    "locked_last_nout": 116, "prediction_access": sys.argv[4] == "observation"
}, indent=2, sort_keys=True) + "\n")
PY

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GFORTRAN_UNBUFFERED_ALL=1
MIN_NOUT=116
MAX_NOUT=116
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
PAIR_ID=${PAIR_ID}
ROLE=${ROLE}
SEED=${SEED}
JOB_ID=${SLURM_JOB_ID}
LAST_NOUT=${LAST_NOUT}
LAST_TIME=${LAST_TIME}
LAST_TUD=${LAST_TUD}
MIN_NOUT=${MIN_NOUT}
MAX_NOUT=${MAX_NOUT}
EOF
}

sync_outputs() {
  local BASE FILE NOUT COMPLETED
  local -a MOMENT_FILES=()
  mapfile -t MOMENT_FILES < <(find "${LOCAL}" -maxdepth 1 -type f -name 'MV11_MOMENTS_NOUT*.DAT' -printf '%f\n' | sort)
  ((${#MOMENT_FILES[@]} > 0)) || return 0
  COMPLETED=$(grep -c 'Output files written' "${LOCAL}/run.log" 2>/dev/null || true)
  (( COMPLETED >= ${#MOMENT_FILES[@]} )) || return 0
  for BASE in "${MOMENT_FILES[@]}"; do
    NOUT=${BASE#MV11_MOMENTS_NOUT}; NOUT=${NOUT%.DAT}; NOUT=$((10#${NOUT}))
    if (( NOUT >= 100 && NOUT <= 116 )); then
      install -m 0644 "${LOCAL}/${BASE}" "${MOMENT_DIR}/${BASE}"
    fi
  done
  FILE=${MOMENT_FILES[-1]}
  NOUT=${FILE#MV11_MOMENTS_NOUT}; NOUT=${NOUT%.DAT}; NOUT=$((10#${NOUT}))
  if (( NOUT <= LAST_NOUT )); then return 0; fi
  LAST_NOUT=${NOUT}
  LAST_TIME=$(awk 'NR == 2 {
    for (i=1; i<=NF; i++) {
      if ($i == "TIME=") { print $(i+1); exit }
      if ($i ~ /^TIME=[^[:space:]]+$/) { sub(/^TIME=/, "", $i); print $i; exit }
    }
  }' "${LOCAL}/${FILE}")
  LAST_TIME=${LAST_TIME:-0}
  LAST_TUD=$(awk -v T="${LAST_TIME}" -v U="${UINF}" -v D="${DIAMETER}" 'BEGIN{printf "%.12g",T*U/D}')
  write_status
  echo "MV17B_SNAPSHOT_PASS case=${CASE_ID} nout=${LAST_NOUT} time=${LAST_TIME} tud=${LAST_TUD}"
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
  (cd "${CASE_DIR}" && find . -type f ! -name 'CASE_SHA256SUMS.txt' -print0 | sort -z | xargs -0 sha256sum > CASE_SHA256SUMS.txt) 2>/dev/null || true
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
DEADLINE=$((SECONDS + 11 * 3600))

while kill -0 "${SOLVER_PID}" 2>/dev/null; do
  sync_outputs
  if (( LAST_NOUT >= MAX_NOUT )); then
    STATUS=LOCKED_WINDOW_COMPLETE
    echo "MV17B_LOCKED_WINDOW_COMPLETE case=${CASE_ID} nout=${LAST_NOUT} tud=${LAST_TUD}"
    exit 0
  fi
  if grep -qiE 'Fortran runtime error|Error termination|Segmentation fault|Floating point exception|FATAL:' run.log 2>/dev/null; then
    STATUS=FATAL
    tail -n 200 run.log || true
    exit 20
  fi
  if (( SECONDS >= DEADLINE )); then STATUS=PARTIAL_WALLTIME; exit 21; fi
  sleep 10
done

wait "${SOLVER_PID}" || SOLVER_RC=$?
SOLVER_RC=${SOLVER_RC:-0}
sync_outputs
if (( LAST_NOUT >= MIN_NOUT )); then
  STATUS=LOCKED_WINDOW_COMPLETE
  exit 0
fi
STATUS=SOLVER_EXITED_BEFORE_LOCKED_WINDOW
if (( SOLVER_RC != 0 )); then
  exit "${SOLVER_RC}"
fi
# A normal solver exit before the locked acquisition window is still a failed
# trajectory.  Never let solver return code zero mark this task COMPLETED.
exit 22
