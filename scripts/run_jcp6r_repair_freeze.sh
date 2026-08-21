#!/usr/bin/env bash
set -Eeuo pipefail

JCP6R_CODE_COMMIT=385d785c436ea93790f62f55032e5b6fa5f2694e
JCP6R_RAW="https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/${JCP6R_CODE_COMMIT}"
JCP6R_WORK=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP6R_MODEL_LOCK
JCP6R_CODE="${JCP6R_WORK}/code"
JCP6R_JCP4=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP4_M8_REFERENCE/JCP4_M8_REFERENCE.zip
JCP6R_M10_ROOT=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc/mv11_ds2v_cylinder_runs
JCP6R_FAILED_JCP6=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP6_MODEL_LOCK/JCP6_MODEL_LOCK.zip
EXPECTED_JCP4_SHA256=b8d8e7f8bc9b0be2027145bade3859d0c3b42e962239c4c3859d6224d1c3cf31
EXPECTED_FAILED_JCP6_SHA256=1f3881fc4b59716922a581304891f5d7b721423b2301d518210b499b54483ac1

trap 'RC=$?; echo "JCP6R_BOOTSTRAP_FAILED rc=${RC} line=${LINENO} command=${BASH_COMMAND}" >&2; exit "${RC}"' ERR

if [[ -f "${JCP6R_WORK}/JCP6R_MODEL_LOCK.zip" && -f "${JCP6R_WORK}/JCP6R_MODEL_LOCK.zip.sha256" ]]; then
  if (cd "${JCP6R_WORK}" && sha256sum -c JCP6R_MODEL_LOCK.zip.sha256) && \
    python - "${JCP6R_WORK}/JCP6R_MODEL_LOCK.zip" <<'PY'
import io
import sys
import zipfile
import numpy as np
with zipfile.ZipFile(sys.argv[1]) as archive:
    with np.load(io.BytesIO(archive.read("JCP6R_MODEL.npz")), allow_pickle=False) as model:
        if not all(
            not np.issubdtype(model[name].dtype, np.number) or np.isfinite(model[name]).all()
            for name in model.files
        ):
            raise SystemExit(1)
PY
  then
    echo "JCP6R_MODEL_LOCK_ALREADY_COMPLETE=1"
    echo "UPLOAD=${JCP6R_WORK}/JCP6R_MODEL_LOCK.zip ${JCP6R_WORK}/JCP6R_MODEL_LOCK.zip.sha256"
    exit 0
  fi
  STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
  mv "${JCP6R_WORK}/JCP6R_MODEL_LOCK.zip" "${JCP6R_WORK}/JCP6R_MODEL_LOCK.zip.nonfinite-${STAMP}"
  mv "${JCP6R_WORK}/JCP6R_MODEL_LOCK.zip.sha256" "${JCP6R_WORK}/JCP6R_MODEL_LOCK.zip.sha256.nonfinite-${STAMP}"
  echo "JCP6R_NONFINITE_ARCHIVE_PRESERVED=${STAMP}"
fi
[[ "$(sha256sum "${JCP6R_JCP4}" | awk '{print $1}')" == "${EXPECTED_JCP4_SHA256}" ]] || { echo "JCP4_CHECKSUM_MISMATCH=1" >&2; exit 2; }
[[ "$(sha256sum "${JCP6R_FAILED_JCP6}" | awk '{print $1}')" == "${EXPECTED_FAILED_JCP6_SHA256}" ]] || { echo "FAILED_JCP6_CHECKSUM_MISMATCH=1" >&2; exit 2; }
[[ -d "${JCP6R_M10_ROOT}/MV11_DS2V_CYLINDER_20260813_170355" ]] || { echo "MISSING_LOCKED_M10_CAMPAIGN=1" >&2; exit 2; }

mkdir -p "${JCP6R_CODE}/scripts" "${JCP6R_CODE}/reference_data/mohammadzadeh_2012" "${JCP6R_WORK}/logs"
FILES=(
  scripts/jcp6_train_freeze.py
  scripts/jcp6r_repair_freeze.py
  scripts/unity_jcp6r_repair_freeze.sbatch
  reference_data/mohammadzadeh_2012/jcp6r_repair_protocol.json
)
for FILE in "${FILES[@]}"; do
  mkdir -p "${JCP6R_CODE}/$(dirname "${FILE}")"
  curl --retry 3 -fsSL "${JCP6R_RAW}/${FILE}" -o "${JCP6R_CODE}/${FILE}"
done
python -m py_compile "${JCP6R_CODE}/scripts/jcp6_train_freeze.py" "${JCP6R_CODE}/scripts/jcp6r_repair_freeze.py"
bash -n "${JCP6R_CODE}/scripts/unity_jcp6r_repair_freeze.sbatch"

JOB_ID="$(sbatch --parsable \
  --output="${JCP6R_WORK}/logs/j6r-repair_%j.out" \
  --error="${JCP6R_WORK}/logs/j6r-repair_%j.err" \
  --export="ALL,JCP6R_CODE=${JCP6R_CODE},JCP6R_WORK=${JCP6R_WORK},JCP6R_JCP4=${JCP6R_JCP4},JCP6R_M10_ROOT=${JCP6R_M10_ROOT},JCP6R_FAILED_JCP6=${JCP6R_FAILED_JCP6}" \
  "${JCP6R_CODE}/scripts/unity_jcp6r_repair_freeze.sbatch")"
printf 'JCP6R_JOB_ID=%q\nJCP6R_WORK=%q\nJCP6R_CODE_COMMIT=%q\n' "${JOB_ID}" "${JCP6R_WORK}" "${JCP6R_CODE_COMMIT}" > "${JCP6R_WORK}/LAST_JCP6R.env"
echo "JCP6R_REPAIR_SUBMITTED=1"
echo "JCP6R_JOB_ID=${JOB_ID}"
echo "MONITOR=squeue -j ${JOB_ID}"
echo "WHEN_COMPLETE_UPLOAD=${JCP6R_WORK}/JCP6R_MODEL_LOCK.zip ${JCP6R_WORK}/JCP6R_MODEL_LOCK.zip.sha256"
