#!/bin/bash
#SBATCH --job-name=j8-m12-ref
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --array=0-3%4

set -Eeuo pipefail
: "${JCP8_SOURCE_DIR:?}"; : "${JCP8_DATA:?}"; : "${JCP8_HEAT_BENCH:?}"
: "${JCP8_WORK:?}"; : "${JCP8_CODE:?}"
SEEDS=(26082801 26082802 26082803 26082804)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
UNIT="${JCP8_WORK}/units/seed_${SEED}"
CASE="${JCP8_WORK}/cases/seed_${SEED}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${UNIT}" "${CASE}"
find "${JCP8_SOURCE_DIR}" -maxdepth 1 -type f -exec cp -p -- {} "${CASE}/" \;
cp -p -- "${JCP8_HEAT_BENCH}" "${CASE}/HEAT-BENCH.TXT"
printf '1\n0\n0\n0\n0\n3\n' > "${CASE}/interactive_input.txt"
python "${JCP8_CODE}/scripts/prepare_jcp8_ds2v_m12.py" --source "${JCP8_SOURCE_DIR}/Plasma_Calculations2.bird_m10_fresh.F90" --data "${JCP8_DATA}" --output-source "${CASE}/DS2V_JCP8_M12.F90" --output-data "${CASE}/DS2VD.DAT" --report "${CASE}/JCP8_PATCH_REPORT.json"
cd "${CASE}"
printf '%s\n' "${SEED}" > RANDOM_SEED.IN
: > RUNIND
gfortran -O2 -g -std=legacy -fdec -fno-automatic -fallow-argument-mismatch -ffree-line-length-none -fno-range-check -fbacktrace -fcheck=bounds DS2V_JCP8_M12.F90 -o ds2v_jcp8_m12
set +e
timeout --signal=TERM --kill-after=60s 1050m ./ds2v_jcp8_m12 < interactive_input.txt > JCP8_DS2V.out 2> JCP8_DS2V.err
RC=$?
set -e
if [[ "${RC}" != 0 && "${RC}" != 124 ]]; then echo "JCP8_DS2V_FAILED=${RC}" >&2; tail -n 160 JCP8_DS2V.err >&2 || true; exit "${RC}"; fi
python "${JCP8_CODE}/scripts/verify_jcp8_m12_reference.py" --case "${CASE}" --seed "${SEED}" --patch-report "${CASE}/JCP8_PATCH_REPORT.json" --minimum-total-blocks 64 --retained-blocks 40 --summary "${UNIT}/JCP8_M12_REFERENCE_SUMMARY.json" --archive "${UNIT}/JCP8_M12_REFERENCE_seed_${SEED}.zip"
echo "JCP8_REFERENCE_UNIT_COMPLETE seed=${SEED}"
