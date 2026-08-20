#!/bin/bash
#SBATCH --job-name=j4-m8-ref
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-3%4

set -Eeuo pipefail
: "${JCP4_SOURCE_DIR:?}"
: "${JCP4_DATA:?}"
: "${JCP4_HEAT_BENCH:?}"
: "${JCP4_WORK:?}"
: "${JCP4_CODE:?}"

SEEDS=(26082401 26082402 26082403 26082404)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
UNIT="${JCP4_WORK}/units/seed_${SEED}"
CASE="${JCP4_WORK}/cases/seed_${SEED}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${UNIT}" "${CASE}"
find "${JCP4_SOURCE_DIR}" -maxdepth 1 -type f -exec cp -p -- {} "${CASE}/" \;
cp -p -- "${JCP4_HEAT_BENCH}" "${CASE}/HEAT-BENCH.TXT"
printf '1\n0\n0\n0\n0\n3\n' > "${CASE}/interactive_input.txt"

python "${JCP4_CODE}/scripts/prepare_jcp4_ds2v_m8.py" \
  --source "${JCP4_SOURCE_DIR}/Plasma_Calculations2.bird_m10_fresh.F90" \
  --data "${JCP4_DATA}" \
  --output-source "${CASE}/DS2V_JCP4_M8.F90" \
  --output-data "${CASE}/DS2VD.DAT" \
  --report "${CASE}/JCP4_PATCH_REPORT.json"

cd "${CASE}"
printf '%s\n' "${SEED}" > RANDOM_SEED.IN
: > RUNIND
gfortran -O2 -g -std=legacy -fdec -fno-automatic -fallow-argument-mismatch \
  -ffree-line-length-none -fno-range-check -fbacktrace -fcheck=bounds \
  DS2V_JCP4_M8.F90 -o ds2v_jcp4_m8

set +e
timeout --signal=TERM --kill-after=60s 330m \
  ./ds2v_jcp4_m8 < interactive_input.txt > JCP4_DS2V.out 2> JCP4_DS2V.err
RC=$?
set -e
if [[ "${RC}" != 0 && "${RC}" != 124 ]]; then
  echo "JCP4_DS2V_FAILED=${RC}" >&2
  tail -n 160 JCP4_DS2V.err >&2 || true
  exit "${RC}"
fi

python "${JCP4_CODE}/scripts/verify_jcp4_m8_reference.py" \
  --case "${CASE}" \
  --seed "${SEED}" \
  --patch-report "${CASE}/JCP4_PATCH_REPORT.json" \
  --minimum-total-blocks 60 \
  --retained-blocks 40 \
  --summary "${UNIT}/JCP4_M8_REFERENCE_SUMMARY.json" \
  --archive "${UNIT}/JCP4_M8_REFERENCE_seed_${SEED}.zip"
touch "${UNIT}/JCP4_M8_REFERENCE_SUCCESS"
echo "JCP4_M8_REFERENCE_UNIT_COMPLETE seed=${SEED}"
