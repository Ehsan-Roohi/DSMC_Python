#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_TARGET="/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc"
TARGET_ROOT="${1:-${MV11_PROJECT_ROOT:-${DEFAULT_TARGET}}}"
PAYLOAD_TARGET="${TARGET_ROOT}/mv11_ds2v_cylinder"
JOB_POINTER="${TARGET_ROOT}/LAST_MV11_DS2V_CYLINDER_JOB.env"

test -d "${TARGET_ROOT}"
test -s "${JOB_POINTER}"
source "${JOB_POINTER}"
: "${MV11_CAMPAIGN_ROOT:?}"
test -d "${MV11_CAMPAIGN_ROOT}/cases"

for relative in \
  patcher/patch_ds2v_mv11.py \
  tools/analyze_mv11_cylinder.py \
  tests/test_mv11_ds2v_cylinder.py \
  case/mv11_cylinder_protocol.json \
  scripts/unity_mv11_post.sbatch; do
  install -D -m 0644 "${BUNDLE_ROOT}/payload/${relative}" "${PAYLOAD_TARGET}/${relative}"
done
chmod +x \
  "${PAYLOAD_TARGET}/patcher/patch_ds2v_mv11.py" \
  "${PAYLOAD_TARGET}/tools/analyze_mv11_cylinder.py" \
  "${PAYLOAD_TARGET}/tests/test_mv11_ds2v_cylinder.py"

python3 "${PAYLOAD_TARGET}/tests/test_mv11_ds2v_cylinder.py"

STATUS_COUNT=$(find "${MV11_CAMPAIGN_ROOT}/cases" -type f -name RUN_STATUS.env | wc -l)
MOMENT_COUNT=$(find "${MV11_CAMPAIGN_ROOT}/cases" -type f -name 'MV11_MOMENTS_NOUT*.DAT' | wc -l)
EXPECTED_MOMENTS="${MV11_RECOVERY_EXPECTED_MOMENT_FILES:-240}"
if [[ "${STATUS_COUNT}" -ne 4 || "${MOMENT_COUNT}" -ne "${EXPECTED_MOMENTS}" ]]; then
  echo "MV11 recovery input mismatch: statuses=${STATUS_COUNT}, moments=${MOMENT_COUNT}, expected=${EXPECTED_MOMENTS}" >&2
  exit 41
fi

POST_JOB_ID=$(sbatch --parsable \
  --export="ALL,MV11_PROJECT_ROOT=${TARGET_ROOT},MV11_CAMPAIGN_ROOT=${MV11_CAMPAIGN_ROOT},MV11_PAYLOAD_ROOT=${PAYLOAD_TARGET}" \
  "${PAYLOAD_TARGET}/scripts/unity_mv11_post.sbatch")

cat > "${TARGET_ROOT}/LAST_MV11_DS2V_CYLINDER_POST_RECOVERY.env" <<EOF
MV11_CAMPAIGN_ROOT=${MV11_CAMPAIGN_ROOT}
MV11_RECOVERY_POST_JOB_ID=${POST_JOB_ID}
MV11_RECOVERY_MOMENT_FILES=${MOMENT_COUNT}
EOF
echo "MV11_POST_RECOVERY_SUBMITTED job=${POST_JOB_ID} campaign=${MV11_CAMPAIGN_ROOT} moments=${MOMENT_COUNT}"
