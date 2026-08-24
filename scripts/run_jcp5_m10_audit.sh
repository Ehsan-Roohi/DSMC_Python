#!/usr/bin/env bash
set -Eeuo pipefail

JCP5_CODE_COMMIT=d0242e5d98248fc7e0924b9506cc09f2c79633b1
JCP5_RAW="https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/${JCP5_CODE_COMMIT}"
JCP5_M10_ROOT=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc/mv11_ds2v_cylinder_runs
JCP5_JCP4_ARCHIVE=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP4_M8_REFERENCE/JCP4_M8_REFERENCE.zip
JCP5_EXPECTED_JCP4_SHA256=b8d8e7f8bc9b0be2027145bade3859d0c3b42e962239c4c3859d6224d1c3cf31
JCP5_WORK=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP5_M10_AUDIT
JCP5_CODE="${JCP5_WORK}/code"

trap 'RC=$?; echo "JCP5_BOOTSTRAP_FAILED rc=${RC} line=${LINENO} command=${BASH_COMMAND}" >&2; exit "${RC}"' ERR

if [[ -f "${JCP5_WORK}/JCP5_M10_AUDIT.zip" && -f "${JCP5_WORK}/JCP5_M10_AUDIT.zip.sha256" ]]; then
  (cd "${JCP5_WORK}" && sha256sum -c JCP5_M10_AUDIT.zip.sha256)
  echo "JCP5_M10_AUDIT_ALREADY_COMPLETE=1"
  echo "UPLOAD=${JCP5_WORK}/JCP5_M10_AUDIT.zip ${JCP5_WORK}/JCP5_M10_AUDIT.zip.sha256"
  exit 0
fi

[[ -d "${JCP5_M10_ROOT}" ]] || { echo "MISSING_M10_ROOT=1" >&2; exit 2; }
[[ -f "${JCP5_JCP4_ARCHIVE}" ]] || { echo "MISSING_JCP4_ARCHIVE=1" >&2; exit 2; }
JCP4_SHA="$(sha256sum "${JCP5_JCP4_ARCHIVE}" | awk '{print $1}')"
[[ "${JCP4_SHA}" == "${JCP5_EXPECTED_JCP4_SHA256}" ]] || { echo "JCP4_CHECKSUM_MISMATCH actual=${JCP4_SHA}" >&2; exit 2; }

mkdir -p "${JCP5_CODE}/scripts" "${JCP5_WORK}/logs"
for FILE in scripts/audit_jcp5_m10_assets.py scripts/unity_jcp5_m10_audit.sbatch; do
  curl --retry 3 -fsSL "${JCP5_RAW}/${FILE}" -o "${JCP5_CODE}/${FILE}"
done
python -m py_compile "${JCP5_CODE}/scripts/audit_jcp5_m10_assets.py"
bash -n "${JCP5_CODE}/scripts/unity_jcp5_m10_audit.sbatch"

JOB_ID="$(sbatch --parsable \
  --output="${JCP5_WORK}/logs/j5-m10-audit_%j.out" \
  --error="${JCP5_WORK}/logs/j5-m10-audit_%j.err" \
  --export="ALL,JCP5_M10_ROOT=${JCP5_M10_ROOT},JCP5_JCP4_ARCHIVE=${JCP5_JCP4_ARCHIVE},JCP5_WORK=${JCP5_WORK},JCP5_CODE=${JCP5_CODE}" \
  "${JCP5_CODE}/scripts/unity_jcp5_m10_audit.sbatch")"
printf 'JCP5_JOB_ID=%q\nJCP5_WORK=%q\nJCP5_CODE_COMMIT=%q\nJCP4_SHA256=%q\n' \
  "${JOB_ID}" "${JCP5_WORK}" "${JCP5_CODE_COMMIT}" "${JCP4_SHA}" > "${JCP5_WORK}/LAST_JCP5.env"
echo "JCP5_M10_AUDIT_SUBMITTED=1"
echo "JCP5_JOB_ID=${JOB_ID}"
echo "MONITOR=squeue -j ${JOB_ID}"
echo "WHEN_COMPLETE_UPLOAD=${JCP5_WORK}/JCP5_M10_AUDIT.zip ${JCP5_WORK}/JCP5_M10_AUDIT.zip.sha256"
