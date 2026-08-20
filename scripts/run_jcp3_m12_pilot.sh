#!/usr/bin/env bash
set -Eeuo pipefail

JCP3_CODE_COMMIT=453bc720cfae225e6334eaca584cae1d4aa31de3
JCP3_RAW="https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/${JCP3_CODE_COMMIT}"
JCP3_SOURCE_DIR=/project/pi_roohie_umass_edu/Ab-initio-shock/ABINITIO_SHOCK_TESTS_v2/DS2V_BIRD_M10_FRESH_ONLY_20260722_234904/source
JCP3_WORK=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP3_M12_PILOT
JCP3_CODE="${JCP3_WORK}/code"
JCP3_SEED=26082301

trap 'RC=$?; echo "JCP3_PILOT_BOOTSTRAP_FAILED rc=${RC} line=${LINENO} command=${BASH_COMMAND}" >&2; exit "${RC}"' ERR

[[ -f "${JCP3_SOURCE_DIR}/Plasma_Calculations2.bird_m10_fresh.F90" ]] || { echo "MISSING_DS2V_SOURCE=1" >&2; exit 2; }
[[ -f "${JCP3_SOURCE_DIR}/DS2VD.DAT" ]] || { echo "MISSING_DS2V_DATA=1" >&2; exit 2; }

if [[ -f "${JCP3_WORK}/JCP3_M12_PILOT.zip" && -f "${JCP3_WORK}/JCP3_M12_PILOT.zip.sha256" ]]; then
  cd "${JCP3_WORK}"
  sha256sum -c JCP3_M12_PILOT.zip.sha256
  echo "JCP3_M12_PILOT_ALREADY_COMPLETE=1"
  echo "UPLOAD=${JCP3_WORK}/JCP3_M12_PILOT.zip ${JCP3_WORK}/JCP3_M12_PILOT.zip.sha256"
  exit 0
fi

mkdir -p "${JCP3_CODE}/scripts" "${JCP3_WORK}/logs"
for FILE in scripts/prepare_jcp3_ds2v_m12.py scripts/verify_jcp3_m12_pilot.py scripts/unity_jcp3_m12_pilot.sbatch; do
  curl --retry 3 -fsSL "${JCP3_RAW}/${FILE}" -o "${JCP3_CODE}/${FILE}"
done
python -m py_compile "${JCP3_CODE}/scripts/prepare_jcp3_ds2v_m12.py" "${JCP3_CODE}/scripts/verify_jcp3_m12_pilot.py"
bash -n "${JCP3_CODE}/scripts/unity_jcp3_m12_pilot.sbatch"

JOB_ID="$(sbatch --parsable \
  --output="${JCP3_WORK}/logs/j3-m12-pilot_%j.out" \
  --error="${JCP3_WORK}/logs/j3-m12-pilot_%j.err" \
  --export="ALL,JCP3_SOURCE_DIR=${JCP3_SOURCE_DIR},JCP3_WORK=${JCP3_WORK},JCP3_CODE=${JCP3_CODE},JCP3_SEED=${JCP3_SEED}" \
  "${JCP3_CODE}/scripts/unity_jcp3_m12_pilot.sbatch")"
printf 'JCP3_PILOT_JOB_ID=%q\nJCP3_WORK=%q\nJCP3_CODE_COMMIT=%q\n' "${JOB_ID}" "${JCP3_WORK}" "${JCP3_CODE_COMMIT}" > "${JCP3_WORK}/LAST_JCP3_PILOT.env"
echo "JCP3_M12_PILOT_SUBMITTED=1"
echo "JCP3_PILOT_JOB_ID=${JOB_ID}"
echo "MONITOR=squeue -j ${JOB_ID}"
echo "WHEN_COMPLETE_UPLOAD=${JCP3_WORK}/JCP3_M12_PILOT.zip ${JCP3_WORK}/JCP3_M12_PILOT.zip.sha256"
