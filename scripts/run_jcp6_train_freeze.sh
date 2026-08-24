#!/usr/bin/env bash
set -Eeuo pipefail

JCP6_CODE_COMMIT=29e64d6151638e038baf9144a1c5e0750afd9419
JCP6_RAW="https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/${JCP6_CODE_COMMIT}"
JCP6_WORK=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP6_MODEL_LOCK
JCP6_CODE="${JCP6_WORK}/code"
JCP6_JCP4=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP4_M8_REFERENCE/JCP4_M8_REFERENCE.zip
JCP6_EXPECTED_JCP4_SHA256=b8d8e7f8bc9b0be2027145bade3859d0c3b42e962239c4c3859d6224d1c3cf31
JCP6_M10_ROOT=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc/mv11_ds2v_cylinder_runs

trap 'RC=$?; echo "JCP6_BOOTSTRAP_FAILED rc=${RC} line=${LINENO} command=${BASH_COMMAND}" >&2; exit "${RC}"' ERR

if [[ -f "${JCP6_WORK}/JCP6_MODEL_LOCK.zip" && -f "${JCP6_WORK}/JCP6_MODEL_LOCK.zip.sha256" ]]; then
  (cd "${JCP6_WORK}" && sha256sum -c JCP6_MODEL_LOCK.zip.sha256)
  echo "JCP6_MODEL_LOCK_ALREADY_COMPLETE=1"
  echo "UPLOAD=${JCP6_WORK}/JCP6_MODEL_LOCK.zip ${JCP6_WORK}/JCP6_MODEL_LOCK.zip.sha256"
  exit 0
fi
[[ -f "${JCP6_JCP4}" ]] || { echo "MISSING_JCP4_ARCHIVE=1" >&2; exit 2; }
[[ "$(sha256sum "${JCP6_JCP4}" | awk '{print $1}')" == "${JCP6_EXPECTED_JCP4_SHA256}" ]] || { echo "JCP4_CHECKSUM_MISMATCH=1" >&2; exit 2; }
[[ -d "${JCP6_M10_ROOT}/MV11_DS2V_CYLINDER_20260813_170355" ]] || { echo "MISSING_LOCKED_M10_CAMPAIGN=1" >&2; exit 2; }

mkdir -p "${JCP6_CODE}/scripts" "${JCP6_CODE}/reference_data/mohammadzadeh_2012" "${JCP6_WORK}/logs"
FILES=(
  scripts/jcp6_train_freeze.py
  scripts/unity_jcp6_train_freeze.sbatch
  reference_data/mohammadzadeh_2012/jcp6_model_lock_protocol.json
)
for FILE in "${FILES[@]}"; do
  mkdir -p "${JCP6_CODE}/$(dirname "${FILE}")"
  curl --retry 3 -fsSL "${JCP6_RAW}/${FILE}" -o "${JCP6_CODE}/${FILE}"
done
python -m py_compile "${JCP6_CODE}/scripts/jcp6_train_freeze.py"
bash -n "${JCP6_CODE}/scripts/unity_jcp6_train_freeze.sbatch"

JOB_ID="$(sbatch --parsable \
  --output="${JCP6_WORK}/logs/j6-train-lock_%j.out" \
  --error="${JCP6_WORK}/logs/j6-train-lock_%j.err" \
  --export="ALL,JCP6_CODE=${JCP6_CODE},JCP6_WORK=${JCP6_WORK},JCP6_JCP4=${JCP6_JCP4},JCP6_M10_ROOT=${JCP6_M10_ROOT}" \
  "${JCP6_CODE}/scripts/unity_jcp6_train_freeze.sbatch")"
printf 'JCP6_JOB_ID=%q\nJCP6_WORK=%q\nJCP6_CODE_COMMIT=%q\n' "${JOB_ID}" "${JCP6_WORK}" "${JCP6_CODE_COMMIT}" > "${JCP6_WORK}/LAST_JCP6.env"
echo "JCP6_MODEL_LOCK_SUBMITTED=1"
echo "JCP6_JOB_ID=${JOB_ID}"
echo "MONITOR=squeue -j ${JOB_ID}"
echo "WHEN_COMPLETE_UPLOAD=${JCP6_WORK}/JCP6_MODEL_LOCK.zip ${JCP6_WORK}/JCP6_MODEL_LOCK.zip.sha256"
