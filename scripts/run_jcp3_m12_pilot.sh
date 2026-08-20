#!/usr/bin/env bash
set -Eeuo pipefail

JCP3_CODE_COMMIT=4ebba81514f77ee6c59b4535f0fbf296ebcfaf28
JCP3_RAW="https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/${JCP3_CODE_COMMIT}"
JCP3_SOURCE_DIR=/project/pi_roohie_umass_edu/Ab-initio-shock/ABINITIO_SHOCK_TESTS_v2/DS2V_BIRD_M10_FRESH_ONLY_20260722_234904/source
JCP3_DATA_SEARCH_ROOT=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc/mv11_ds2v_cylinder_runs
JCP3_EXPECTED_DATA_SHA256=a13e82650ffa7a0303b0353ad385b198839c2c738df7cff98ce343806e736b96
JCP3_EXPECTED_HEAT_BENCH_SHA256=2d94da3d86786afd1c497994cad935cfca1d188d9431bf16960fbc533e3f6c34
JCP3_WORK=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP3_M12_PILOT
JCP3_CODE="${JCP3_WORK}/code"
JCP3_SEED=26082301

trap 'RC=$?; echo "JCP3_PILOT_BOOTSTRAP_FAILED rc=${RC} line=${LINENO} command=${BASH_COMMAND}" >&2; exit "${RC}"' ERR

[[ -f "${JCP3_SOURCE_DIR}/Plasma_Calculations2.bird_m10_fresh.F90" ]] || { echo "MISSING_DS2V_SOURCE=1" >&2; exit 2; }

JCP3_DATA=
for CANDIDATE in \
  "${JCP3_SOURCE_DIR}/DS2VD.DAT" \
  "${JCP3_SOURCE_DIR}/../DS2VD.DAT" \
  "${JCP3_DATA_SEARCH_ROOT}/MV11_DS2V_CYLINDER_20260813_170355/input/DS2VD.DAT" \
  "${JCP3_DATA_SEARCH_ROOT}/MV11_DS2V_CYLINDER_20260813_133511/input/DS2VD.DAT"
do
  if [[ -f "${CANDIDATE}" ]]; then
    JCP3_DATA="${CANDIDATE}"
    break
  fi
done
if [[ -z "${JCP3_DATA}" && -d "${JCP3_DATA_SEARCH_ROOT}" ]]; then
  JCP3_DATA="$(find "${JCP3_DATA_SEARCH_ROOT}" -type f -path '*/input/DS2VD.DAT' -print -quit 2>/dev/null || true)"
fi
[[ -n "${JCP3_DATA}" && -f "${JCP3_DATA}" ]] || { echo "MISSING_DS2V_DATA=1 search_root=${JCP3_DATA_SEARCH_ROOT}" >&2; exit 2; }
JCP3_DATA_SHA256="$(sha256sum "${JCP3_DATA}" | awk '{print $1}')"
[[ "${JCP3_DATA_SHA256}" == "${JCP3_EXPECTED_DATA_SHA256}" ]] || {
  echo "DS2V_DATA_CHECKSUM_MISMATCH expected=${JCP3_EXPECTED_DATA_SHA256} actual=${JCP3_DATA_SHA256} path=${JCP3_DATA}" >&2
  exit 2
}
echo "JCP3_DATA=${JCP3_DATA}"
echo "JCP3_DATA_SHA256=${JCP3_DATA_SHA256}"

JCP3_HEAT_BENCH=
for CANDIDATE in \
  "$(dirname "${JCP3_DATA}")/HEAT-BENCH.TXT" \
  "${JCP3_DATA_SEARCH_ROOT}/MV11_DS2V_CYLINDER_20260813_170355/input/HEAT-BENCH.TXT" \
  "${JCP3_DATA_SEARCH_ROOT}/MV11_DS2V_CYLINDER_20260813_133511/input/HEAT-BENCH.TXT"
do
  if [[ -f "${CANDIDATE}" ]]; then
    JCP3_HEAT_BENCH="${CANDIDATE}"
    break
  fi
done
if [[ -z "${JCP3_HEAT_BENCH}" && -d "${JCP3_DATA_SEARCH_ROOT}" ]]; then
  JCP3_HEAT_BENCH="$(find "${JCP3_DATA_SEARCH_ROOT}" -type f -path '*/input/HEAT-BENCH.TXT' -print -quit 2>/dev/null || true)"
fi
[[ -n "${JCP3_HEAT_BENCH}" && -f "${JCP3_HEAT_BENCH}" ]] || { echo "MISSING_HEAT_BENCH=1 search_root=${JCP3_DATA_SEARCH_ROOT}" >&2; exit 2; }
JCP3_HEAT_BENCH_SHA256="$(sha256sum "${JCP3_HEAT_BENCH}" | awk '{print $1}')"
[[ "${JCP3_HEAT_BENCH_SHA256}" == "${JCP3_EXPECTED_HEAT_BENCH_SHA256}" ]] || {
  echo "HEAT_BENCH_CHECKSUM_MISMATCH expected=${JCP3_EXPECTED_HEAT_BENCH_SHA256} actual=${JCP3_HEAT_BENCH_SHA256} path=${JCP3_HEAT_BENCH}" >&2
  exit 2
}
echo "JCP3_HEAT_BENCH=${JCP3_HEAT_BENCH}"
echo "JCP3_HEAT_BENCH_SHA256=${JCP3_HEAT_BENCH_SHA256}"

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
  --export="ALL,JCP3_SOURCE_DIR=${JCP3_SOURCE_DIR},JCP3_DATA=${JCP3_DATA},JCP3_HEAT_BENCH=${JCP3_HEAT_BENCH},JCP3_WORK=${JCP3_WORK},JCP3_CODE=${JCP3_CODE},JCP3_SEED=${JCP3_SEED}" \
  "${JCP3_CODE}/scripts/unity_jcp3_m12_pilot.sbatch")"
printf 'JCP3_PILOT_JOB_ID=%q\nJCP3_WORK=%q\nJCP3_CODE_COMMIT=%q\nJCP3_DATA=%q\nJCP3_DATA_SHA256=%q\nJCP3_HEAT_BENCH=%q\nJCP3_HEAT_BENCH_SHA256=%q\n' \
  "${JOB_ID}" "${JCP3_WORK}" "${JCP3_CODE_COMMIT}" "${JCP3_DATA}" "${JCP3_DATA_SHA256}" \
  "${JCP3_HEAT_BENCH}" "${JCP3_HEAT_BENCH_SHA256}" > "${JCP3_WORK}/LAST_JCP3_PILOT.env"
echo "JCP3_M12_PILOT_SUBMITTED=1"
echo "JCP3_PILOT_JOB_ID=${JOB_ID}"
echo "MONITOR=squeue -j ${JOB_ID}"
echo "WHEN_COMPLETE_UPLOAD=${JCP3_WORK}/JCP3_M12_PILOT.zip ${JCP3_WORK}/JCP3_M12_PILOT.zip.sha256"
