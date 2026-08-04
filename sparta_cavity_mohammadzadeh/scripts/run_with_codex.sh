#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if ! command -v codex >/dev/null 2>&1; then
  echo "Codex CLI is not installed. Follow the official install link in README.md." >&2
  exit 2
fi

cd "${ROOT_DIR}"
mkdir -p runs
codex exec \
  --sandbox workspace-write \
  --output-last-message runs/codex_smoke_report.md \
  - < codex_prompt_run_sparta.md

echo "Codex report: ${ROOT_DIR}/runs/codex_smoke_report.md"
