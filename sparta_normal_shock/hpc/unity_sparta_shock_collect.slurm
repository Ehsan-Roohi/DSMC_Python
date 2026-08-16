#!/usr/bin/env bash
#SBATCH --job-name=sp-shock-pack
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sp-shock-pack-%j.out
#SBATCH --error=logs/sp-shock-pack-%j.err

set -euo pipefail

ROOT_DIR="${SPARTA_CASE_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "${ROOT_DIR}" && -d "${ROOT_DIR}" ]] || exit 2
ROOT_DIR="$(cd "${ROOT_DIR}" && pwd)"
ARRAY_JOB_ID="${SPARTA_ARRAY_JOB_ID:?Missing SPARTA_ARRAY_JOB_ID}"
BUILD_JOB_ID="${SPARTA_BUILD_JOB_ID:?Missing SPARTA_BUILD_JOB_ID}"
SEEDS=(20260803 20260819 20260831)
MACH_LABELS=(M25 M30 M50)
MACH_VALUES=(2.5 3.0 5.0)
FINAL_DUMP_NAME="profile.final.00160000"
MANIFEST_REL="runs/SPARTA_NORMAL_SHOCK_RESULTS_${ARRAY_JOB_ID}.manifest.txt"
BUNDLE_REL="runs/SPARTA_NORMAL_SHOCK_RESULTS_${ARRAY_JOB_ID}.tar.gz"
MANIFEST="${ROOT_DIR}/${MANIFEST_REL}"
BUNDLE="${ROOT_DIR}/${BUNDLE_REL}"
INCLUDE_PATHS=("${MANIFEST_REL}")
COMPLETE_COUNT=0

mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/runs"
{
  printf 'SPARTA steady normal-shock return bundle\n'
  printf 'configuration=Mach_2.5_3_5_x_3_seeds_nx600_ppc64_average160000\n'
  printf 'alignment=unsmoothed_density_midpoint_translation\n'
  printf 'spatial_smoothing=none\n'
  printf 'build_job_id=%s\n' "${BUILD_JOB_ID}"
  printf 'array_job_id=%s\n' "${ARRAY_JOB_ID}"
  printf 'collector_job_id=%s\n' "${SLURM_JOB_ID}"
  printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${MANIFEST}"

for index in "${!MACH_LABELS[@]}"; do
  label="${MACH_LABELS[index]}"
  ensemble_dir="${ROOT_DIR}/runs/ensemble_${ARRAY_JOB_ID}_${label}"
  complete_dirs=()
  for seed in "${SEEDS[@]}"; do
    run_rel="runs/shock_${ARRAY_JOB_ID}_${label}_seed_${seed}"
    run_dir="${ROOT_DIR}/${run_rel}"
    if [[ -s "${run_dir}/${FINAL_DUMP_NAME}" && -s "${run_dir}/profile_normalized.csv" ]] && \
       grep -q '^status=complete$' "${run_dir}/unity_run_metadata.txt" 2>/dev/null; then
      printf '%s_seed_%s=complete\n' "${label}" "${seed}" >> "${MANIFEST}"
      INCLUDE_PATHS+=("${run_rel}")
      complete_dirs+=("${run_dir}")
      COMPLETE_COUNT=$((COMPLETE_COUNT + 1))
    elif [[ -d "${run_dir}" ]]; then
      printf '%s_seed_%s=incomplete\n' "${label}" "${seed}" >> "${MANIFEST}"
      INCLUDE_PATHS+=("${run_rel}")
    else
      printf '%s_seed_%s=missing\n' "${label}" "${seed}" >> "${MANIFEST}"
    fi
  done
  if (( ${#complete_dirs[@]} >= 2 )); then
    python3 "${ROOT_DIR}/scripts/postprocess.py" ensemble \
      --output "${ensemble_dir}" "${complete_dirs[@]}" \
      > "${ensemble_dir}.output.json"
    mv "${ensemble_dir}.output.json" "${ensemble_dir}/postprocess_output.json"
    INCLUDE_PATHS+=("${ensemble_dir#"${ROOT_DIR}/"}")
    printf '%s_ensemble_count=%s\n' "${label}" "${#complete_dirs[@]}" >> "${MANIFEST}"
  else
    printf '%s_ensemble_count=%s\n' "${label}" "${#complete_dirs[@]}" >> "${MANIFEST}"
  fi
done

shopt -s nullglob
for log_path in \
  "${ROOT_DIR}"/logs/sp-shock-build-"${BUILD_JOB_ID}".* \
  "${ROOT_DIR}"/logs/sp-shock-prod-"${ARRAY_JOB_ID}"_*.out \
  "${ROOT_DIR}"/logs/sp-shock-prod-"${ARRAY_JOB_ID}"_*.err; do
  INCLUDE_PATHS+=("${log_path#"${ROOT_DIR}/"}")
done
printf 'complete_member_count=%s\n' "${COMPLETE_COUNT}" >> "${MANIFEST}"
tar --exclude='*/restart.shock.1' --exclude='*/restart.shock.2' \
  -czf "${BUNDLE}" -C "${ROOT_DIR}" "${INCLUDE_PATHS[@]}"
tar -tzf "${BUNDLE}" >/dev/null
sha256sum "${BUNDLE}" > "${BUNDLE}.sha256"
echo "RETURN THIS FILE: ${BUNDLE}"
echo "SHA256 FILE: ${BUNDLE}.sha256"
echo "Complete production members: ${COMPLETE_COUNT}/9"
(( COMPLETE_COUNT == 9 )) || { echo "Bundle created, but the array is incomplete." >&2; exit 6; }

