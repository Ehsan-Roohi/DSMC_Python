#!/usr/bin/env bash
#SBATCH --job-name=unity-watch
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=00:05:00
#SBATCH --nice=10000

set -u

APP_ROOT="${UNITY_WATCHTOWER_APP_ROOT:?UNITY_WATCHTOWER_APP_ROOT is required}"
STATE_ROOT="${UNITY_WATCHTOWER_STATE_ROOT:-/project/pi_roohie_umass_edu/UNITY_MONITOR/scheduler}"
INTERVAL_MINUTES="${UNITY_WATCHTOWER_INTERVAL_MINUTES:-15}"
CONFIG_PATH="${UNITY_WATCHTOWER_CONFIG:-${HOME}/.config/unity-watchtower/config.json}"
STOP_FILE="${STATE_ROOT}/STOP"
STATE_FILE="${STATE_ROOT}/watcher_job.env"
LOCK_FILE="${STATE_ROOT}/collector.lock"
LOG_DIR="${STATE_ROOT}/../logs"

mkdir -p "${STATE_ROOT}" "${LOG_DIR}"

schedule_next() {
  if [[ -f "${STOP_FILE}" ]]; then
    printf 'STOP file is present; no successor will be submitted.\n'
    return 0
  fi
  next_raw="$(sbatch --parsable \
    --begin="now+${INTERVAL_MINUTES}minutes" \
    --export=ALL,UNITY_WATCHTOWER_APP_ROOT="${APP_ROOT}",UNITY_WATCHTOWER_STATE_ROOT="${STATE_ROOT}",UNITY_WATCHTOWER_INTERVAL_MINUTES="${INTERVAL_MINUTES}",UNITY_WATCHTOWER_CONFIG="${CONFIG_PATH}" \
    --output="${LOG_DIR}/watchtower-%j.out" \
    --error="${LOG_DIR}/watchtower-%j.err" \
    "${APP_ROOT}/hpc/watch_once.slurm")" || {
      printf 'WARNING: successor submission failed; run scripts/start_watcher.sh again.\n' >&2
      return 0
    }
  next_id="${next_raw%%;*}"
  printf 'WATCHTOWER_JOB_ID=%q\n' "${next_id}" > "${STATE_FILE}"
  printf 'WATCHTOWER_SUBMITTED_AT=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${STATE_FILE}"
  printf 'Scheduled next collector job %s in %s minutes.\n' "${next_id}" "${INTERVAL_MINUTES}"
}
trap schedule_next EXIT

if [[ -f "${STOP_FILE}" ]]; then
  printf 'STOP file is present; collector skipped.\n'
  exit 0
fi

if command -v flock >/dev/null 2>&1; then
  flock -n "${LOCK_FILE}" python3 "${APP_ROOT}/monitor.py" --config "${CONFIG_PATH}" run --push || status=$?
else
  python3 "${APP_ROOT}/monitor.py" --config "${CONFIG_PATH}" run --push || status=$?
fi
status="${status:-0}"
if [[ "${status}" -ne 0 ]]; then
  printf 'Collector exited with status %s; the chain will continue.\n' "${status}" >&2
fi
exit 0
