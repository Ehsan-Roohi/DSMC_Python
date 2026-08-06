#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_ROOT="${UNITY_WATCHTOWER_STATE_ROOT:-/project/pi_roohie_umass_edu/UNITY_MONITOR/scheduler}"
LOG_DIR="${STATE_ROOT}/../logs"
STATE_FILE="${STATE_ROOT}/watcher_job.env"
INTERVAL_MINUTES="${UNITY_WATCHTOWER_INTERVAL_MINUTES:-15}"
CONFIG_PATH="${UNITY_WATCHTOWER_CONFIG:-${HOME}/.config/unity-watchtower/config.json}"
mkdir -p "${STATE_ROOT}" "${LOG_DIR}"

if [[ -f "${STATE_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${STATE_FILE}"
  if [[ -n "${WATCHTOWER_JOB_ID:-}" ]] && squeue -h -j "${WATCHTOWER_JOB_ID}" | grep -q .; then
    printf 'Unity Watchtower is already scheduled as job %s.\n' "${WATCHTOWER_JOB_ID}"
    exit 0
  fi
fi

rm -f "${STATE_ROOT}/STOP"
job_raw="$(sbatch --parsable \
  --export=ALL,UNITY_WATCHTOWER_APP_ROOT="${APP_ROOT}",UNITY_WATCHTOWER_STATE_ROOT="${STATE_ROOT}",UNITY_WATCHTOWER_INTERVAL_MINUTES="${INTERVAL_MINUTES}",UNITY_WATCHTOWER_CONFIG="${CONFIG_PATH}" \
  --output="${LOG_DIR}/watchtower-%j.out" \
  --error="${LOG_DIR}/watchtower-%j.err" \
  "${APP_ROOT}/hpc/watch_once.slurm")"
job_id="${job_raw%%;*}"
printf 'WATCHTOWER_JOB_ID=%q\n' "${job_id}" > "${STATE_FILE}"
printf 'WATCHTOWER_SUBMITTED_AT=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${STATE_FILE}"
printf 'Started Unity Watchtower job %s; interval=%s minutes.\n' "${job_id}" "${INTERVAL_MINUTES}"
