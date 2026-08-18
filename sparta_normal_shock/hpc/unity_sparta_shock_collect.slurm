#!/usr/bin/env bash
#SBATCH --job-name=sp2-shock-pack
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sp2-shock-pack-%j.out
#SBATCH --error=logs/sp2-shock-pack-%j.err

set -euo pipefail

ROOT_DIR="${SPARTA_CASE_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -n "${ROOT_DIR}" && -d "${ROOT_DIR}" ]] || exit 2
ROOT_DIR="$(cd "${ROOT_DIR}" && pwd)"
ARRAY_JOB_ID="${SPARTA_V2_ARRAY_JOB_ID:?Missing SPARTA_V2_ARRAY_JOB_ID}"
BUILD_JOB_ID="${SPARTA_V2_BUILD_JOB_ID:?Missing SPARTA_V2_BUILD_JOB_ID}"
SEEDS=(20260803 20260819 20260831)
MACH_LABELS=(M25 M30 M50)
FINAL_DUMP_NAME="profile.final.00320000"
MANIFEST_REL="runs/SPARTA_NORMAL_SHOCK_V2_RESULTS_${ARRAY_JOB_ID}.manifest.txt"
BUNDLE_REL="runs/SPARTA_NORMAL_SHOCK_V2_RESULTS_${ARRAY_JOB_ID}.tar.gz"
MANIFEST="${ROOT_DIR}/${MANIFEST_REL}"
BUNDLE="${ROOT_DIR}/${BUNDLE_REL}"
INCLUDE_PATHS=("${MANIFEST_REL}")
EXECUTED_COUNT=0
VALIDATED_COUNT=0
ENSEMBLE_FAILURE_COUNT=0

mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/runs"
{
  printf 'SPARTA steady normal-shock v2 return bundle\n'
  printf 'configuration=Mach_2.5_3_5_x_3_seeds_halfspan30_nx1200_ppc64_warmup80000_average320000\n'
  printf 'alignment=unsmoothed_density_midpoint_translation\n'
  printf 'spatial_smoothing=none\n'
  printf 'far_field_relative_error_limit=0.03\n'
  printf 'flux_relative_error_from_upstream_limit=0.005\n'
  printf 'plateau_relative_slope_per_lambda_limit=0.005\n'
  printf 'checkpoint_profile_rms_relative_change_limit=0.02\n'
  printf 'checkpoint_profile_max_relative_change_limit=0.05\n'
  printf 'ensemble_ci=pointwise_student_t_df2_critical_4.30265272991\n'
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
    run_status=""
    if [[ -f "${run_dir}/unity_run_metadata.txt" ]]; then
      run_status="$(sed -n 's/^status=//p' "${run_dir}/unity_run_metadata.txt" | tail -1)"
    fi
    if [[ "${run_status}" == "complete" || "${run_status}" == "physics_failed" ]]; then
      EXECUTED_COUNT=$((EXECUTED_COUNT + 1))
    fi
    if [[ -s "${run_dir}/${FINAL_DUMP_NAME}" && -s "${run_dir}/profile_normalized.csv" && \
          -s "${run_dir}/validation_metrics.json" && "${run_status}" == "complete" ]] && \
       grep -Fxq 'physics_gate_pass=true' "${run_dir}/unity_run_metadata.txt" 2>/dev/null && \
       python3 -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1], encoding="utf-8")).get("physics_gate_pass") is True else 1)' \
         "${run_dir}/validation_metrics.json"; then
      printf '%s_seed_%s=validated\n' "${label}" "${seed}" >> "${MANIFEST}"
      INCLUDE_PATHS+=("${run_rel}")
      complete_dirs+=("${run_dir}")
      VALIDATED_COUNT=$((VALIDATED_COUNT + 1))
    elif [[ "${run_status}" == "physics_failed" || -s "${run_dir}/validation_metrics.json" ]]; then
      printf '%s_seed_%s=physics_failed\n' "${label}" "${seed}" >> "${MANIFEST}"
      INCLUDE_PATHS+=("${run_rel}")
    elif [[ -d "${run_dir}" ]]; then
      printf '%s_seed_%s=incomplete\n' "${label}" "${seed}" >> "${MANIFEST}"
      INCLUDE_PATHS+=("${run_rel}")
    else
      printf '%s_seed_%s=missing\n' "${label}" "${seed}" >> "${MANIFEST}"
    fi
  done
  if (( ${#complete_dirs[@]} == 3 )); then
    ensemble_stdout_rel="logs/sp2-shock-ensemble-${ARRAY_JOB_ID}-${label}.out"
    ensemble_stderr_rel="logs/sp2-shock-ensemble-${ARRAY_JOB_ID}-${label}.err"
    set +e
    python3 "${ROOT_DIR}/scripts/postprocess.py" ensemble \
      --output "${ensemble_dir}" "${complete_dirs[@]}" \
      > "${ROOT_DIR}/${ensemble_stdout_rel}" 2> "${ROOT_DIR}/${ensemble_stderr_rel}"
    ensemble_status=$?
    set -e
    if (( ensemble_status == 0 )) && [[ -d "${ensemble_dir}" ]]; then
      mv "${ROOT_DIR}/${ensemble_stdout_rel}" "${ensemble_dir}/postprocess_output.json"
      INCLUDE_PATHS+=("${ensemble_dir#"${ROOT_DIR}/"}" "${ensemble_stderr_rel}")
      printf '%s_ensemble_status=complete\n' "${label}" >> "${MANIFEST}"
      printf '%s_ensemble_count=%s\n' "${label}" "${#complete_dirs[@]}" >> "${MANIFEST}"
    else
      ENSEMBLE_FAILURE_COUNT=$((ENSEMBLE_FAILURE_COUNT + 1))
      INCLUDE_PATHS+=("${ensemble_stdout_rel}" "${ensemble_stderr_rel}")
      [[ -d "${ensemble_dir}" ]] && INCLUDE_PATHS+=("${ensemble_dir#"${ROOT_DIR}/"}")
      printf '%s_ensemble_status=failed_exit_%s\n' "${label}" "${ensemble_status}" >> "${MANIFEST}"
      printf '%s_ensemble_count=0\n' "${label}" >> "${MANIFEST}"
    fi
  else
    printf '%s_ensemble_status=not_created\n' "${label}" >> "${MANIFEST}"
    printf '%s_ensemble_count=%s\n' "${label}" "${#complete_dirs[@]}" >> "${MANIFEST}"
  fi
done

shopt -s nullglob
for log_path in \
  "${ROOT_DIR}"/logs/sp2-shock-build-"${BUILD_JOB_ID}"* \
  "${ROOT_DIR}"/logs/sp2-shock-prod-"${ARRAY_JOB_ID}"_*.out \
  "${ROOT_DIR}"/logs/sp2-shock-prod-"${ARRAY_JOB_ID}"_*.err; do
  INCLUDE_PATHS+=("${log_path#"${ROOT_DIR}/"}")
done
BUILD_READY="${ROOT_DIR}/logs/BUILD_READY_${BUILD_JOB_ID}.txt"
[[ -f "${BUILD_READY}" ]] && INCLUDE_PATHS+=("${BUILD_READY#"${ROOT_DIR}/"}")
printf 'executed_member_count=%s\n' "${EXECUTED_COUNT}" >> "${MANIFEST}"
printf 'validated_member_count=%s\n' "${VALIDATED_COUNT}" >> "${MANIFEST}"
printf 'complete_member_count=%s\n' "${VALIDATED_COUNT}" >> "${MANIFEST}"
printf 'ensemble_failure_count=%s\n' "${ENSEMBLE_FAILURE_COUNT}" >> "${MANIFEST}"
tar --exclude='*/restart.shock.1' --exclude='*/restart.shock.2' \
  -czf "${BUNDLE}" -C "${ROOT_DIR}" "${INCLUDE_PATHS[@]}"
tar -tzf "${BUNDLE}" >/dev/null
sha256sum "${BUNDLE}" > "${BUNDLE}.sha256"
echo "RETURN THIS FILE: ${BUNDLE}"
echo "SHA256 FILE: ${BUNDLE}.sha256"
echo "Executed production members: ${EXECUTED_COUNT}/9"
echo "Physics-validated production members: ${VALIDATED_COUNT}/9"
(( VALIDATED_COUNT == 9 && ENSEMBLE_FAILURE_COUNT == 0 )) || {
  echo "DIAGNOSTIC BUNDLE CREATED; VALIDATION INCOMPLETE." >&2
  exit 6
}
echo "PUBLICATION-READY BUNDLE: all 9 members passed the declared physics gates."
