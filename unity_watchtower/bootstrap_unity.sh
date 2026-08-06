#!/usr/bin/env bash
set -euo pipefail

INSTALL_ROOT="${UNITY_WATCHTOWER_INSTALL_ROOT:-/project/pi_roohie_umass_edu/UNITY_MONITOR}"
CHECKOUT_ROOT="${INSTALL_ROOT}/source"
APP_ROOT="${CHECKOUT_ROOT}/unity_watchtower"
CONFIG_DIR="${HOME}/.config/unity-watchtower"
CONFIG_PATH="${CONFIG_DIR}/config.json"
SOURCE_REPO="${UNITY_WATCHTOWER_SOURCE_REPO:-https://github.com/Ehsan-Roohi/DSMC_Python.git}"
SOURCE_BRANCH="${UNITY_WATCHTOWER_SOURCE_BRANCH:-agent/unity-watchtower}"

mkdir -p "${INSTALL_ROOT}" "${CONFIG_DIR}" "${HOME}/bin"
if [[ ! -d "${CHECKOUT_ROOT}/.git" ]]; then
  if [[ -e "${CHECKOUT_ROOT}" ]]; then
    printf 'Installation source path exists but is not a Git checkout: %s\n' "${CHECKOUT_ROOT}" >&2
    exit 2
  fi
  git clone --depth 1 --branch "${SOURCE_BRANCH}" "${SOURCE_REPO}" "${CHECKOUT_ROOT}"
else
  git -C "${CHECKOUT_ROOT}" pull --ff-only origin "${SOURCE_BRANCH}"
fi
if [[ ! -d "${APP_ROOT}" ]]; then
  printf 'unity_watchtower directory is missing from source branch.\n' >&2
  exit 2
fi

chmod 700 "${APP_ROOT}/bin/unity-watch" "${APP_ROOT}/scripts/"*.sh
chmod 700 "${APP_ROOT}/bootstrap_unity.sh"
chmod 700 "${APP_ROOT}/hpc/watch_once.slurm"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  install -m 600 "${APP_ROOT}/config/config.default.json" "${CONFIG_PATH}"
  printf 'Installed default project registry at %s.\n' "${CONFIG_PATH}"
else
  printf 'Preserved existing project registry at %s.\n' "${CONFIG_PATH}"
fi
ln -sfn "${APP_ROOT}/bin/unity-watch" "${HOME}/bin/unity-watch"

"${APP_ROOT}/scripts/configure_github.sh"
python3 "${APP_ROOT}/monitor.py" --config "${CONFIG_PATH}" run --push
"${APP_ROOT}/scripts/start_watcher.sh"

printf '\nUnity Watchtower installation is complete.\n'
printf 'Dashboard repository: https://github.com/Ehsan-Roohi/UnityMonitor\n'
printf 'One-shot status:       %s\n' "${HOME}/bin/unity-watch show"
printf 'Scheduler status:      %s\n' "${APP_ROOT}/scripts/watcher_status.sh"
printf 'Stop automation:       %s\n' "${APP_ROOT}/scripts/stop_watcher.sh"
