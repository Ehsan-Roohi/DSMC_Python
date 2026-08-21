#!/bin/bash
#SBATCH --job-name=j7-m12-eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=09:00:00
#SBATCH --array=0-11%8

set -Eeuo pipefail
: "${JCP7_SOURCE_DIR:?}"
: "${JCP7_DATA:?}"
: "${JCP7_HEAT_BENCH:?}"
: "${JCP7_WORK:?}"
: "${JCP7_CODE:?}"

SEEDS=(26082701 26082702 26082703 26082704 26082705 26082706 26082707 26082708 26082709 26082710 26082711 26082712)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
UNIT="${JCP7_WORK}/units/seed_${SEED}"
CASE="${JCP7_WORK}/cases/seed_${SEED}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${UNIT}" "${CASE}"
find "${JCP7_SOURCE_DIR}" -maxdepth 1 -type f -exec cp -p -- {} "${CASE}/" \;
cp -p -- "${JCP7_HEAT_BENCH}" "${CASE}/HEAT-BENCH.TXT"
printf '1\n0\n0\n0\n0\n3\n' > "${CASE}/interactive_input.txt"

python "${JCP7_CODE}/scripts/prepare_jcp3_ds2v_m12.py" \
  --source "${JCP7_SOURCE_DIR}/Plasma_Calculations2.bird_m10_fresh.F90" \
  --data "${JCP7_DATA}" \
  --output-source "${CASE}/DS2V_JCP7_M12.F90" \
  --output-data "${CASE}/DS2VD.DAT" \
  --report "${CASE}/JCP7_PATCH_REPORT.json"
cd "${CASE}"
printf '%s\n' "${SEED}" > RANDOM_SEED.IN
: > RUNIND
gfortran -O2 -g -std=legacy -fdec -fno-automatic -fallow-argument-mismatch \
  -ffree-line-length-none -fno-range-check -fbacktrace -fcheck=bounds \
  DS2V_JCP7_M12.F90 -o ds2v_jcp7_m12
set +e
timeout --signal=TERM --kill-after=60s 520m \
  ./ds2v_jcp7_m12 < interactive_input.txt > JCP7_DS2V.out 2> JCP7_DS2V.err
RC=$?
set -e
if [[ "${RC}" != 0 && "${RC}" != 124 ]]; then
  echo "JCP7_DS2V_FAILED=${RC}" >&2
  tail -n 160 JCP7_DS2V.err >&2 || true
  exit "${RC}"
fi
python "${JCP7_CODE}/scripts/verify_jcp7_m12_evaluation.py" \
  --case "${CASE}" --seed "${SEED}" --patch-report "${CASE}/JCP7_PATCH_REPORT.json" \
  --minimum-total-blocks 40 --retained-blocks 14 \
  --summary "${UNIT}/JCP7_M12_EVALUATION_SUMMARY.json" \
  --archive "${UNIT}/JCP7_M12_EVALUATION_seed_${SEED}.zip"
echo "JCP7_EVALUATION_CANDIDATE_COMPLETE seed=${SEED}"
