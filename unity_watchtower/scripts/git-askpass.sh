#!/usr/bin/env bash
set -euo pipefail

TOKEN_FILE="${UNITY_MONITOR_TOKEN_FILE:-${HOME}/.config/unity-watchtower/github.token}"
case "${1:-}" in
  *Username*|*username*)
    printf '%s\n' "${UNITY_MONITOR_GITHUB_USERNAME:-Ehsan-Roohi}"
    ;;
  *)
    if [[ ! -r "${TOKEN_FILE}" ]]; then
      printf '%s\n' "Unity Watchtower token file is not readable: ${TOKEN_FILE}" >&2
      exit 1
    fi
    IFS= read -r token < "${TOKEN_FILE}"
    printf '%s\n' "${token}"
    ;;
esac
