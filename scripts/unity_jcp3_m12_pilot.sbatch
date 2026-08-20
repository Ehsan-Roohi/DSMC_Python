#!/bin/bash
#SBATCH --job-name=j3-m12-pilot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00

set -Eeuo pipefail
: "${JCP3_SOURCE_DIR:?}"
: "${JCP3_WORK:?}"
: "${JCP3_CODE:?}"
: "${JCP3_SEED:?}"

CASE="${JCP3_WORK}/case_seed_${JCP3_SEED}"
mkdir -p "${CASE}"
find "${JCP3_SOURCE_DIR}" -maxdepth 1 -type f -exec cp -p -- {} "${CASE}/" \;
printf '1\n0\n0\n0\n0\n3\n' > "${CASE}/interactive_input.txt"

python "${JCP3_CODE}/scripts/prepare_jcp3_ds2v_m12.py" \
  --source "${JCP3_SOURCE_DIR}/Plasma_Calculations2.bird_m10_fresh.F90" \
  --data "${JCP3_SOURCE_DIR}/DS2VD.DAT" \
  --output-source "${CASE}/DS2V_JCP3_M12.F90" \
  --output-data "${CASE}/DS2VD.DAT" \
  --report "${CASE}/JCP3_PATCH_REPORT.json"

cd "${CASE}"
printf '%s\n' "${JCP3_SEED}" > RANDOM_SEED.IN
: > JCP3_PILOT_ONLY
: > RUNIND
gfortran -O2 -g -std=legacy -fdec -fno-automatic -fallow-argument-mismatch \
  -ffree-line-length-none -fno-range-check -fbacktrace -fcheck=bounds \
  DS2V_JCP3_M12.F90 -o ds2v_jcp3_m12

set +e
timeout --signal=TERM --kill-after=60s 02:30:00 \
  ./ds2v_jcp3_m12 < interactive_input.txt > JCP3_DS2V.out 2> JCP3_DS2V.err
RC=$?
set -e
if [[ "${RC}" != 0 && "${RC}" != 124 ]]; then
  echo "JCP3_DS2V_FAILED=${RC}" >&2
  exit "${RC}"
fi

python "${JCP3_CODE}/scripts/verify_jcp3_m12_pilot.py" \
  --case "${CASE}" \
  --patch-report "${CASE}/JCP3_PATCH_REPORT.json" \
  --archive "${JCP3_WORK}/JCP3_M12_PILOT.zip"
