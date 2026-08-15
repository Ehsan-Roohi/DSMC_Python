#!/usr/bin/env bash
#SBATCH --job-name=moh-mv17b-a1-run
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=12:00:00

set -euo pipefail
: "${MV17B_OUTPUT_ROOT:?}"
: "${SLURM_ARRAY_TASK_ID:?}"
case ",1,2,3,4,9,10,11," in
  *",${SLURM_ARRAY_TASK_ID},"*) ;;
  *) echo "MV17B_A1_REFUSE_UNDECLARED_TASK=${SLURM_ARRAY_TASK_ID}" >&2; exit 91 ;;
esac

CAMPAIGN="${MV17B_OUTPUT_ROOT}/campaign"
test -s "${MV17B_OUTPUT_ROOT}/MV17B_A1_MECHANICAL_RECOVERY_LOCK.json"
test ! -e "${MV17B_OUTPUT_ROOT}/analysis"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MV17B_OUTPUT_ROOT}/seed_table.tsv")
IFS=$'\t' read -r CASE_ID PAIR_ID ROLE SEED <<< "${LINE}"
test -n "${CASE_ID}" && test -n "${PAIR_ID}" && test -n "${ROLE}" && test -n "${SEED}"

CASE_DIR="${CAMPAIGN}/cases/${CASE_ID}"
RESULT_DIR="${CASE_DIR}/results"
ORIGINAL_MOMENTS="${RESULT_DIR}/moments"
RECOVERY_DIR="${CAMPAIGN}/recovery/${CASE_ID}"
STAGED_MOMENTS="${RECOVERY_DIR}/moments"
LOG_DIR="${RECOVERY_DIR}/logs"
test -s "${RESULT_DIR}/RUN_STATUS.env"
grep -qx 'STATUS=SOLVER_EXITED_BEFORE_LOCKED_WINDOW' "${RESULT_DIR}/RUN_STATUS.env"
test ! -e "${RESULT_DIR}/MV17B_A1_RECOVERY_PASS"
test ! -e "${RECOVERY_DIR}"
mkdir -p "${STAGED_MOMENTS}" "${LOG_DIR}"
exec > >(tee "${LOG_DIR}/recovery_${SLURM_JOB_ID}.out") 2>&1

LOCAL_BASE="${TMPDIR:-/tmp}"
LOCAL="${LOCAL_BASE}/mv17b_a1_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if [[ "${LOCAL}" != "${LOCAL_BASE}"/mv17b_a1_* ]]; then
  echo "MV17B-A1 unsafe local scratch path" >&2
  exit 90
fi
mkdir -p "${LOCAL}"
install -m 0755 "${CAMPAIGN}/build/ds2v_bird_mv17b" "${LOCAL}/ds2v"
install -m 0644 "${CAMPAIGN}/input/DS2VD.DAT" "${LOCAL}/DS2VD.DAT"
install -m 0644 "${CAMPAIGN}/input/HEAT-BENCH.TXT" "${LOCAL}/HEAT-BENCH.TXT"
ln -s HEAT-BENCH.TXT "${LOCAL}/HEAT-BENCH.txt"
mapfile -t CONTROL < "${CAMPAIGN}/input/interactive_input.txt"
test "${#CONTROL[@]}" -eq 6
test "${CONTROL[5]}" = "3"
CONTROL[5]=4
printf '%s\n' "${CONTROL[@]}" > "${LOCAL}/interactive_input.txt"
printf '%s\n' "${SEED}" > "${LOCAL}/RANDOM_SEED.IN"

ORIGINAL_LAST=$(awk -F= '$1=="LAST_NOUT"{print $2}' "${RESULT_DIR}/RUN_STATUS.env")
ORIGINAL_TUD=$(awk -F= '$1=="LAST_TUD"{print $2}' "${RESULT_DIR}/RUN_STATUS.env")
cat > "${RECOVERY_DIR}/MV17B_A1_RECOVERY_METADATA.json" <<EOF
{
  "case_id": "${CASE_ID}",
  "pair_id": "${PAIR_ID}",
  "role": "${ROLE}",
  "seed": ${SEED},
  "array_index": ${SLURM_ARRAY_TASK_ID},
  "job_id": ${SLURM_JOB_ID},
  "original_last_nout": ${ORIGINAL_LAST},
  "original_last_tU_over_D": ${ORIGINAL_TUD},
  "original_IRUN": 3,
  "recovery_IRUN": 4,
  "same_seed": true,
  "same_executable": true,
  "same_DS2VD_deck": true,
  "overlap_must_be_byte_identical": true,
  "scientific_contract_changed": false
}
EOF
sha256sum "${CAMPAIGN}/build/ds2v_bird_mv17b" "${CAMPAIGN}/input/DS2VD.DAT" \
  "${LOCAL}/ds2v" "${LOCAL}/DS2VD.DAT" > "${RECOVERY_DIR}/INPUT_EXECUTABLE_SHA256.txt"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 GFORTRAN_UNBUFFERED_ALL=1
SOLVER_PID=""
LAST_NOUT=0
sync_staged() {
  local FILE BASE NOUT COMPLETED
  local -a FILES=()
  mapfile -t FILES < <(find "${LOCAL}" -maxdepth 1 -type f -name 'MV11_MOMENTS_NOUT*.DAT' -printf '%f\n' | sort)
  ((${#FILES[@]} > 0)) || return 0
  COMPLETED=$(grep -c 'Output files written' "${LOCAL}/run.log" 2>/dev/null || true)
  (( COMPLETED >= ${#FILES[@]} )) || return 0
  for BASE in "${FILES[@]}"; do
    NOUT=${BASE#MV11_MOMENTS_NOUT}; NOUT=${NOUT%.DAT}; NOUT=$((10#${NOUT}))
    if (( NOUT >= 100 && NOUT <= 116 )); then
      install -m 0644 "${LOCAL}/${BASE}" "${STAGED_MOMENTS}/${BASE}"
    fi
    (( NOUT > LAST_NOUT )) && LAST_NOUT=${NOUT}
  done
  echo "MV17B_A1_RECOVERY_PROGRESS case=${CASE_ID} last_nout=${LAST_NOUT}"
}
cleanup() {
  set +e
  if [[ -n "${SOLVER_PID}" ]] && kill -0 "${SOLVER_PID}" 2>/dev/null; then
    kill -TERM "${SOLVER_PID}" 2>/dev/null || true
    sleep 3
    kill -KILL "${SOLVER_PID}" 2>/dev/null || true
  fi
  [[ -n "${SOLVER_PID}" ]] && wait "${SOLVER_PID}" 2>/dev/null || true
  sync_staged
  [[ -s "${LOCAL}/run.log" ]] && cp -a "${LOCAL}/run.log" "${RECOVERY_DIR}/run.log"
  [[ -s "${LOCAL}/RNG_SEED_USED.txt" ]] && cp -a "${LOCAL}/RNG_SEED_USED.txt" "${RECOVERY_DIR}/RNG_SEED_USED.txt"
}
trap cleanup EXIT

cd "${LOCAL}"
./ds2v < interactive_input.txt > run.log 2>&1 &
SOLVER_PID=$!
DEADLINE=$((SECONDS + 11 * 3600))
while kill -0 "${SOLVER_PID}" 2>/dev/null; do
  sync_staged
  if [[ -s "${STAGED_MOMENTS}/MV11_MOMENTS_NOUT0116.DAT" ]]; then
    kill -TERM "${SOLVER_PID}" 2>/dev/null || true
    wait "${SOLVER_PID}" 2>/dev/null || true
    SOLVER_PID=""
    break
  fi
  if grep -qiE 'Fortran runtime error|Error termination|Segmentation fault|Floating point exception|FATAL:' run.log 2>/dev/null; then
    tail -n 200 run.log || true
    exit 20
  fi
  if (( SECONDS >= DEADLINE )); then exit 21; fi
  sleep 10
done
if [[ -n "${SOLVER_PID}" ]]; then
  wait "${SOLVER_PID}" || SOLVER_RC=$?
  SOLVER_PID=""
  test "${SOLVER_RC:-0}" -eq 0
fi
sync_staged
for NOUT in $(seq 100 116); do
  FILE=$(printf 'MV11_MOMENTS_NOUT%04d.DAT' "${NOUT}")
  test -s "${STAGED_MOMENTS}/${FILE}"
done

OVERLAP_COUNT=0
for ORIGINAL in "${ORIGINAL_MOMENTS}"/MV11_MOMENTS_NOUT*.DAT; do
  test -f "${ORIGINAL}"
  BASE=$(basename "${ORIGINAL}")
  cmp -s "${ORIGINAL}" "${STAGED_MOMENTS}/${BASE}" || {
    echo "MV17B_A1_OVERLAP_MISMATCH case=${CASE_ID} file=${BASE}" >&2
    exit 31
  }
  OVERLAP_COUNT=$((OVERLAP_COUNT + 1))
done
test "${OVERLAP_COUNT}" -ge 1

# Only absent files are added; original files are never overwritten.
for STAGED in "${STAGED_MOMENTS}"/MV11_MOMENTS_NOUT*.DAT; do
  DESTINATION="${ORIGINAL_MOMENTS}/$(basename "${STAGED}")"
  if [[ -e "${DESTINATION}" ]]; then
    cmp -s "${DESTINATION}" "${STAGED}"
  else
    TEMPORARY="${DESTINATION}.mv17b_a1_${SLURM_JOB_ID}.tmp"
    install -m 0644 "${STAGED}" "${TEMPORARY}"
    mv "${TEMPORARY}" "${DESTINATION}"
  fi
done

test ! -e "${RESULT_DIR}/RUN_STATUS.pre_mv17b_a1.env"
cp -a "${RESULT_DIR}/RUN_STATUS.env" "${RESULT_DIR}/RUN_STATUS.pre_mv17b_a1.env"
TIME116=$(awk 'NR == 2 {for(i=1;i<=NF;i++){if($i=="TIME="){print $(i+1);exit}; if($i~/^TIME=/){sub(/^TIME=/,"",$i);print $i;exit}}}' \
  "${ORIGINAL_MOMENTS}/MV11_MOMENTS_NOUT0116.DAT")
TIME116=${TIME116:-0}
TUD116=$(awk -v T="${TIME116}" 'BEGIN{printf "%.12g",T*2634.1/0.3048}')
cat > "${RESULT_DIR}/RUN_STATUS.env.tmp" <<EOF
STATUS=LOCKED_WINDOW_COMPLETE
CASE_ID=${CASE_ID}
PAIR_ID=${PAIR_ID}
ROLE=${ROLE}
SEED=${SEED}
JOB_ID=${SLURM_JOB_ID}
LAST_NOUT=116
LAST_TIME=${TIME116}
LAST_TUD=${TUD116}
MIN_NOUT=116
MAX_NOUT=116
RECOVERY_STAGE=MV17B_A1
RECOVERY_SAME_SEED=true
RECOVERY_IRUN=4
ORIGINAL_LAST_NOUT=${ORIGINAL_LAST}
OVERLAP_BLOCKS_BYTE_IDENTICAL=${OVERLAP_COUNT}
EOF
mv "${RESULT_DIR}/RUN_STATUS.env.tmp" "${RESULT_DIR}/RUN_STATUS.env"
cp -a "${RECOVERY_DIR}/MV17B_A1_RECOVERY_METADATA.json" "${RESULT_DIR}/MV17B_A1_RECOVERY_METADATA.json"
printf 'MV17B_A1_RECOVERY_PASS case=%s same_seed=%s overlap=%s original_last=%s recovered_last=116\n' \
  "${CASE_ID}" "${SEED}" "${OVERLAP_COUNT}" "${ORIGINAL_LAST}" \
  > "${RESULT_DIR}/MV17B_A1_RECOVERY_PASS"
(cd "${CASE_DIR}" && find . -type f ! -name 'CASE_SHA256SUMS.txt' -print0 | sort -z | xargs -0 sha256sum > CASE_SHA256SUMS.txt)
trap - EXIT
cleanup
echo "MV17B_A1_RECOVERY_PASS case=${CASE_ID} seed=${SEED} original_last=${ORIGINAL_LAST} recovered_last=116 overlap=${OVERLAP_COUNT}"
