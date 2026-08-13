#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_TARGET="/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc"
TARGET_ROOT="${1:-${MV11_PROJECT_ROOT:-${DEFAULT_TARGET}}}"
PAYLOAD_TARGET="${TARGET_ROOT}/mv11_ds2v_cylinder"

test -d "${TARGET_ROOT}"
for relative in \
  patcher/patch_ds2v_mv11.py \
  tools/analyze_mv11_cylinder.py \
  tests/test_mv11_ds2v_cylinder.py \
  case/DS2VD_M10_ARGON_CYLINDER.template \
  case/interactive_input.txt \
  case/HEAT-BENCH.TXT \
  case/mv11_cylinder_protocol.json \
  scripts/unity_mv11_prepare.sbatch \
  scripts/unity_mv11_run_array.sbatch \
  scripts/unity_mv11_post.sbatch \
  scripts/submit_mv11_ds2v_cylinder_unity.sh; do
  install -D -m 0644 "${BUNDLE_ROOT}/payload/${relative}" "${PAYLOAD_TARGET}/${relative}"
done
chmod +x \
  "${PAYLOAD_TARGET}/patcher/patch_ds2v_mv11.py" \
  "${PAYLOAD_TARGET}/tools/analyze_mv11_cylinder.py" \
  "${PAYLOAD_TARGET}/tests/test_mv11_ds2v_cylinder.py" \
  "${PAYLOAD_TARGET}/scripts/submit_mv11_ds2v_cylinder_unity.sh"

python3 "${PAYLOAD_TARGET}/tests/test_mv11_ds2v_cylinder.py"
MV11_PROJECT_ROOT="${TARGET_ROOT}" exec bash "${PAYLOAD_TARGET}/scripts/submit_mv11_ds2v_cylinder_unity.sh"
