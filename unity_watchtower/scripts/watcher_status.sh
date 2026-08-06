#!/usr/bin/env bash
set -euo pipefail

STATE_ROOT="${UNITY_WATCHTOWER_STATE_ROOT:-/project/pi_roohie_umass_edu/UNITY_MONITOR/scheduler}"
STATE_FILE="${STATE_ROOT}/watcher_job.env"
if [[ -f "${STATE_ROOT}/STOP" ]]; then
  printf 'Scheduler: STOPPED\n'
else
  printf 'Scheduler: ENABLED\n'
fi
if [[ ! -f "${STATE_FILE}" ]]; then
  printf 'No watcher job has been recorded.\n'
  exit 1
fi
# shellcheck disable=SC1090
source "${STATE_FILE}"
printf 'Recorded job: %s\n' "${WATCHTOWER_JOB_ID:-unknown}"
if [[ -n "${WATCHTOWER_JOB_ID:-}" ]]; then
  squeue -j "${WATCHTOWER_JOB_ID}" -o '%.18i %.16j %.10T %.12M %.30R' || true
  sacct -X -j "${WATCHTOWER_JOB_ID}" --format=JobID,JobName,State,ExitCode,Elapsed,NodeList || true
fi
