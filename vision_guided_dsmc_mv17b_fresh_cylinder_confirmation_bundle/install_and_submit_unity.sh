#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BUNDLE_ROOT}/.." && pwd)"
DEFAULT_TARGET="/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc"
TARGET_ROOT="${1:-${MV17B_PROJECT_ROOT:-${DEFAULT_TARGET}}}"
PAYLOAD_TARGET="${TARGET_ROOT}/mv17b_fresh_cylinder_confirmation"
MV11_BUNDLE="${REPO_ROOT}/vision_guided_dsmc_mv11_ds2v_cylinder_bundle"
MV16A_BUNDLE="${REPO_ROOT}/vision_guided_dsmc_mv16a_frozen_cylinder_transfer_bundle"
MV17A_BUNDLE="${REPO_ROOT}/vision_guided_dsmc_mv17a_cylinder_native_crossfit_bundle"

if [[ ! -d "${TARGET_ROOT}" || ! -f "${TARGET_ROOT}/vgdsmc/mohammadzadeh_architecture_screen.py" ]]; then
  echo "MV17B_INSTALL_ERROR: target is not the canonical vision_guided_dsmc project: ${TARGET_ROOT}" >&2
  exit 2
fi
for prerequisite in \
  "${MV11_BUNDLE}/payload/patcher/patch_ds2v_mv11.py" \
  "${MV11_BUNDLE}/payload/case/DS2VD_M10_ARGON_CYLINDER.template" \
  "${MV16A_BUNDLE}/payload/vgdsmc/mohammadzadeh_mv16a_frozen_cylinder_transfer.py" \
  "${MV17A_BUNDLE}/payload/vgdsmc/mohammadzadeh_mv17a_cylinder_native_crossfit.py"; do
  test -s "${prerequisite}"
done

# Install the exact source patcher/deck and analysis modules that MV17B imports;
# sibling installers are intentionally not invoked because they submit jobs.
for relative in \
  patcher/patch_ds2v_mv11.py \
  case/DS2VD_M10_ARGON_CYLINDER.template \
  case/interactive_input.txt \
  case/HEAT-BENCH.TXT; do
  install -D -m 0644 "${MV11_BUNDLE}/payload/${relative}" "${TARGET_ROOT}/mv11_ds2v_cylinder/${relative}"
done
install -D -m 0644 \
  "${MV16A_BUNDLE}/payload/vgdsmc/mohammadzadeh_mv16a_frozen_cylinder_transfer.py" \
  "${TARGET_ROOT}/vgdsmc/mohammadzadeh_mv16a_frozen_cylinder_transfer.py"
install -D -m 0644 \
  "${MV16A_BUNDLE}/payload/reference_data/mohammadzadeh_2012/mv16a_frozen_cylinder_transfer_protocol.json" \
  "${TARGET_ROOT}/reference_data/mohammadzadeh_2012/mv16a_frozen_cylinder_transfer_protocol.json"
install -D -m 0644 \
  "${MV17A_BUNDLE}/payload/vgdsmc/mohammadzadeh_mv17a_cylinder_native_crossfit.py" \
  "${TARGET_ROOT}/vgdsmc/mohammadzadeh_mv17a_cylinder_native_crossfit.py"
install -D -m 0644 \
  "${MV17A_BUNDLE}/payload/reference_data/mohammadzadeh_2012/mv17a_cylinder_native_crossfit_protocol.json" \
  "${TARGET_ROOT}/reference_data/mohammadzadeh_2012/mv17a_cylinder_native_crossfit_protocol.json"

for relative in \
  vgdsmc/mohammadzadeh_mv17b_fresh_cylinder_confirmation.py \
  tests/test_mohammadzadeh_mv17b_fresh_cylinder_confirmation.py \
  reference_data/mohammadzadeh_2012/mv17b_fresh_cylinder_confirmation_protocol.json \
  scripts/submit_mohammadzadeh_mv17b_unity.sh \
  scripts/unity_mohammadzadeh_mv17b_prepare.sbatch \
  scripts/unity_mohammadzadeh_mv17b_run_array.sbatch \
  scripts/unity_mohammadzadeh_mv17b_analyze.sbatch \
  scripts/unity_mohammadzadeh_mv17b_post.sbatch; do
  install -D -m 0644 "${BUNDLE_ROOT}/payload/${relative}" "${PAYLOAD_TARGET}/${relative}"
done
install -D -m 0644 \
  "${BUNDLE_ROOT}/payload/vgdsmc/mohammadzadeh_mv17b_fresh_cylinder_confirmation.py" \
  "${TARGET_ROOT}/vgdsmc/mohammadzadeh_mv17b_fresh_cylinder_confirmation.py"
install -D -m 0644 \
  "${BUNDLE_ROOT}/payload/reference_data/mohammadzadeh_2012/mv17b_fresh_cylinder_confirmation_protocol.json" \
  "${TARGET_ROOT}/reference_data/mohammadzadeh_2012/mv17b_fresh_cylinder_confirmation_protocol.json"
chmod +x \
  "${TARGET_ROOT}/mv11_ds2v_cylinder/patcher/patch_ds2v_mv11.py" \
  "${PAYLOAD_TARGET}/tests/test_mohammadzadeh_mv17b_fresh_cylinder_confirmation.py" \
  "${PAYLOAD_TARGET}/scripts/submit_mohammadzadeh_mv17b_unity.sh" \
  "${PAYLOAD_TARGET}/scripts/unity_mohammadzadeh_mv17b_prepare.sbatch" \
  "${PAYLOAD_TARGET}/scripts/unity_mohammadzadeh_mv17b_run_array.sbatch" \
  "${PAYLOAD_TARGET}/scripts/unity_mohammadzadeh_mv17b_analyze.sbatch" \
  "${PAYLOAD_TARGET}/scripts/unity_mohammadzadeh_mv17b_post.sbatch"

PYTHON_BIN="${MV17B_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" && -s "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV17A_CYLINDER_NATIVE_JOB.env" ]]; then
  source "${TARGET_ROOT}/LAST_MOHAMMADZADEH_MV17A_CYLINDER_NATIVE_JOB.env"
  PYTHON_BIN="${MV17A_PYTHON:-}"
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "MV17B_INSTALL_ERROR: no usable Python was found; set MV17B_PYTHON" >&2
  exit 3
fi
if ! "${PYTHON_BIN}" -c 'import matplotlib,numpy,scipy' >/dev/null 2>&1; then
  echo "MV17B_INSTALL_ERROR: selected Python lacks matplotlib, numpy, or scipy: ${PYTHON_BIN}" >&2
  exit 4
fi

cd "${TARGET_ROOT}"
export PYTHONPATH="${TARGET_ROOT}:${PYTHONPATH:-}"
"${PYTHON_BIN}" -m py_compile \
  vgdsmc/mohammadzadeh_mv16a_frozen_cylinder_transfer.py \
  vgdsmc/mohammadzadeh_mv17a_cylinder_native_crossfit.py \
  vgdsmc/mohammadzadeh_mv17b_fresh_cylinder_confirmation.py
"${PYTHON_BIN}" "${PAYLOAD_TARGET}/tests/test_mohammadzadeh_mv17b_fresh_cylinder_confirmation.py"
MV17B_PROJECT_ROOT="${TARGET_ROOT}" MV17B_PYTHON="${PYTHON_BIN}" \
  exec bash "${PAYLOAD_TARGET}/scripts/submit_mohammadzadeh_mv17b_unity.sh"

