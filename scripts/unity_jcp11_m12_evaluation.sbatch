#!/bin/bash
#SBATCH --job-name=j11-m12-eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=09:00:00
#SBATCH --array=0-3%4

set -Eeuo pipefail
: "${JCP11_SOURCE_DIR:?}"
: "${JCP11_DATA:?}"
: "${JCP11_HEAT_BENCH:?}"
: "${JCP11_WORK:?}"
: "${JCP11_CODE:?}"
: "${JCP11_RULE_LOCK:?}"

SEEDS=(26082901 26082902 26082903 26082904)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
UNIT="${JCP11_WORK}/units/seed_${SEED}"
CASE="${JCP11_WORK}/cases/seed_${SEED}_job_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${UNIT}" "${CASE}"

# This assertion is evaluated inside every fresh trajectory job.
[[ -f "${JCP11_RULE_LOCK}" ]] || { echo "MISSING_PRE_RUN_RULE_LOCK=1" >&2; exit 2; }
RULE_SHA256="$(sha256sum "${JCP11_RULE_LOCK}" | awk '{print $1}')"
find "${JCP11_SOURCE_DIR}" -maxdepth 1 -type f -exec cp -p -- {} "${CASE}/" \;
cp -p -- "${JCP11_HEAT_BENCH}" "${CASE}/HEAT-BENCH.TXT"
printf '1\n0\n0\n0\n0\n3\n' > "${CASE}/interactive_input.txt"

python "${JCP11_CODE}/scripts/prepare_jcp3_ds2v_m12.py" \
  --source "${JCP11_SOURCE_DIR}/Plasma_Calculations2.bird_m10_fresh.F90" \
  --data "${JCP11_DATA}" \
  --output-source "${CASE}/DS2V_JCP11_M12.F90" \
  --output-data "${CASE}/DS2VD.DAT" \
  --report "${CASE}/JCP11_PATCH_REPORT.json"
cd "${CASE}"
printf '%s\n' "${SEED}" > RANDOM_SEED.IN
printf '%s  %s\n' "${RULE_SHA256}" "${JCP11_RULE_LOCK}" > JCP11_PRE_RUN_RULE_LOCK.sha256
: > RUNIND
gfortran -O2 -g -std=legacy -fdec -fno-automatic -fallow-argument-mismatch \
  -ffree-line-length-none -fno-range-check -fbacktrace -fcheck=bounds \
  DS2V_JCP11_M12.F90 -o ds2v_jcp11_m12
set +e
timeout --signal=TERM --kill-after=60s 520m \
  ./ds2v_jcp11_m12 < interactive_input.txt > JCP11_DS2V.out 2> JCP11_DS2V.err
RC=$?
set -e
if [[ "${RC}" != 0 && "${RC}" != 124 ]]; then
  echo "JCP11_DS2V_FAILED=${RC}" >&2
  tail -n 160 JCP11_DS2V.err >&2 || true
  exit "${RC}"
fi
python "${JCP11_CODE}/scripts/jcp11_support_validation.py" verify-eval \
  --case "${CASE}" --seed "${SEED}" \
  --patch-report "${CASE}/JCP11_PATCH_REPORT.json" \
  --summary "${UNIT}/JCP11_M12_EVALUATION_SUMMARY.json" \
  --archive "${UNIT}/JCP11_M12_EVALUATION_seed_${SEED}.zip"
echo "JCP11_FRESH_M12_UNIT_COMPLETE seed=${SEED}"
