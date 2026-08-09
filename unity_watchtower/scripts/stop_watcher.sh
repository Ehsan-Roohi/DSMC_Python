#!/usr/bin/env bash
set -euo pipefail

STATE_ROOT="${UNITY_WATCHTOWER_STATE_ROOT:-/project/pi_roohie_umass_edu/UNITY_MONITOR/scheduler}"
STATE_FILE="${STATE_ROOT}/watcher_job.env"
mkdir -p "${STATE_ROOT}"
touch "${STATE_ROOT}/STOP"
if [[ -f "${STATE_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${STATE_FILE}"
  if [[ -n "${WATCHTOWER_JOB_ID:-}" ]]; then
    scancel "${WATCHTOWER_JOB_ID}" 2>/dev/null || true
    printf 'Stopped Unity Watchtower job %s.\n' "${WATCHTOWER_JOB_ID}"
  fi
fi
printf 'Automatic resubmission is disabled until start_watcher.sh is run again.\n'
