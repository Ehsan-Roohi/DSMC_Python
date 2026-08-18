#!/usr/bin/env bash
#SBATCH --job-name=sp-mm-pack
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sp-mm-pack-%j.out
#SBATCH --error=logs/sp-mm-pack-%j.err

set -euo pipefail

ROOT_DIR="${SPARTA_CASE_ROOT:-${SLURM_SUBMIT_DIR:-}}"
RESULTS_BASE="${SPARTA_RESULTS_BASE:?SPARTA_RESULTS_BASE was not exported}"
BUNDLE_DIR="${SPARTA_BUNDLE_DIR:?SPARTA_BUNDLE_DIR was not exported}"
RUN_JOB_ID="${SPARTA_RUN_JOB_ID:?SPARTA_RUN_JOB_ID was not exported}"
BUILD_JOB_ID="${SPARTA_BUILD_JOB_ID:?SPARTA_BUILD_JOB_ID was not exported}"
SEED="${MAXWELL_SINGLE_SEED:-104729}"
if [[ -z "${ROOT_DIR}" || ! -d "${ROOT_DIR}" ]]; then
  echo "SPARTA_CASE_ROOT/SLURM_SUBMIT_DIR does not identify the code directory." >&2
  exit 2
fi
ROOT_DIR="$(cd "${ROOT_DIR}" && pwd)"
CAMPAIGN_ROOT="${RESULTS_BASE}/campaign_${RUN_JOB_ID}"
BUNDLE="${BUNDLE_DIR}/SPARTA_MAXWELL_KNGU005_020_JFM_${RUN_JOB_ID}_SEED${SEED}_TO_ANALYZE.zip"
SHA_FILE="${BUNDLE}.sha256.txt"
CASE_TAGS=(005 020)
KN_VALUES=(0.05 0.20)

mkdir -p "${CAMPAIGN_ROOT}" "${BUNDLE_DIR}"
if [[ -e "${BUNDLE}" || -e "${SHA_FILE}" ]]; then
  echo "Refusing to overwrite an existing return bundle: ${BUNDLE}" >&2
  exit 3
fi

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/sparta-maxwell-pack.XXXXXX")"
trap 'rm -rf "${STAGE}"' EXIT
mkdir -p "${STAGE}/runs" "${STAGE}/provenance/scripts" \
  "${STAGE}/provenance/hpc" "${STAGE}/provenance/tests" \
  "${STAGE}/scheduler_logs"

COMPLETE_COUNT=0
for index in "${!CASE_TAGS[@]}"; do
  case_tag="${CASE_TAGS[index]}"
  kn_gu="${KN_VALUES[index]}"
  source_dir="${CAMPAIGN_ROOT}/kngu${case_tag}_seed_${SEED}"
  target_dir="${STAGE}/runs/kngu${case_tag}_seed_${SEED}"
  mkdir -p "${target_dir}"

  if python3 "${ROOT_DIR}/scripts/validate_jfm_maxwell_kngu_case.py" \
      "${source_dir}" --kn-gu "${kn_gu}" --require-final && \
     grep -q '^status=complete$' "${source_dir}/unity_run_metadata.txt" 2>/dev/null; then
    COMPLETE_COUNT=$((COMPLETE_COUNT + 1))
  fi

  for file_name in \
    case_metadata.json unity_run_metadata.txt in.cavity argon.species maxwell.vss \
    grid.final.00200000 log.cavity mpi_preflight.out mpi_preflight.err \
    screen.out screen.err; do
    if [[ -f "${source_dir}/${file_name}" ]]; then
      cp -- "${source_dir}/${file_name}" "${target_dir}/"
    fi
  done
done

for file_name in \
  generate_jfm_maxwell_kngu020_case.py validate_jfm_maxwell_kngu_case.py; do
  cp -- "${ROOT_DIR}/scripts/${file_name}" "${STAGE}/provenance/scripts/"
done
for file_name in \
  unity_sparta_maxwell_kngu005_020_jfm_build.slurm \
  unity_sparta_maxwell_kngu005_020_jfm_single.slurm \
  unity_sparta_maxwell_kngu005_020_jfm_collect.slurm \
  bootstrap_unity_sparta_maxwell_kngu005_020_jfm.sh; do
  cp -- "${ROOT_DIR}/hpc/${file_name}" "${STAGE}/provenance/hpc/"
done
for test_path in "${ROOT_DIR}"/tests/test_*maxwell*.py; do
  [[ -f "${test_path}" ]] && cp -- "${test_path}" "${STAGE}/provenance/tests/"
done
cp -- "${ROOT_DIR}/MAXWELL_KNGU005_020_JFM.md" "${STAGE}/provenance/"

shopt -s nullglob
for log_path in \
  "${ROOT_DIR}"/logs/sp-mm-build-"${BUILD_JOB_ID}".* \
  "${ROOT_DIR}"/logs/sp-mm-jfm-"${RUN_JOB_ID}"_*.out \
  "${ROOT_DIR}"/logs/sp-mm-jfm-"${RUN_JOB_ID}"_*.err \
  "${ROOT_DIR}"/logs/sp-mm-pack-"${SLURM_JOB_ID}".*; do
  cp -- "${log_path}" "${STAGE}/scheduler_logs/"
done

{
  printf 'SPARTA Maxwell-VSS matched-model JFM return bundle\n'
  printf 'kn_gu_cases=0.05,0.20\n'
  printf 'kn_convention=gu_lambda_over_L\n'
  printf 'collision_model=VSS approximation to IPL Maxwell molecules\n'
  printf 'omega=1\nalpha=2.140\n'
  printf 'diameter_vhs_viscosity_equivalent_m=4.6326659042208623e-10\n'
  printf 'diameter_vss_input_m=4.661368788251916e-10\n'
  printf 'configuration=N160_ppc256_one_seed_warmup40000_samples_per_cell20000\n'
  printf 'dump_schema=maxwell_vss_antifourier_v1_15_fields\n'
  printf 'sonine_B1_role=diagnostic_only_instantaneous_COM\n'
  printf 'quantitative_R_or_Delta_claim_ready=false\n'
  printf 'build_job_id=%s\nrun_array_job_id=%s\ncollector_job_id=%s\n' \
    "${BUILD_JOB_ID}" "${RUN_JOB_ID}" "${SLURM_JOB_ID}"
  printf 'seed=%s\n' "${SEED}"
  printf 'complete_case_count=%s\n' "${COMPLETE_COUNT}"
  printf 'code_commit=%s\n' "$(git -C "${ROOT_DIR}/.." rev-parse HEAD)"
  printf 'sparta_commit=%s\n' "$(git -C "${ROOT_DIR}/third_party/sparta" rev-parse HEAD)"
  printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${STAGE}/CAMPAIGN_MANIFEST.txt"

(
  cd "${STAGE}"
  find . -type f ! -name FILES.sha256 -print0 | LC_ALL=C sort -z | \
    xargs -0 sha256sum > FILES.sha256
)

python3 - "${STAGE}" "${BUNDLE}" <<'PY'
import pathlib
import sys
import zipfile

stage = pathlib.Path(sys.argv[1])
bundle = pathlib.Path(sys.argv[2])
with zipfile.ZipFile(
    bundle, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True
) as archive:
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(stage))
with zipfile.ZipFile(bundle) as archive:
    bad = archive.testzip()
    if bad is not None:
        raise SystemExit(f"ZIP integrity failure at {bad}")
PY
(
  cd "${BUNDLE_DIR}"
  sha256sum "$(basename "${BUNDLE}")" > "$(basename "${SHA_FILE}")"
)

echo "RETURN THIS FILE: ${BUNDLE}"
echo "SHA256 FILE: ${SHA_FILE}"
echo "Complete cases: ${COMPLETE_COUNT}/${#CASE_TAGS[@]}"

if (( COMPLETE_COUNT != ${#CASE_TAGS[@]} )); then
  echo "A diagnostic ZIP was created, but one or both matched-model cases are incomplete." >&2
  exit 6
fi
