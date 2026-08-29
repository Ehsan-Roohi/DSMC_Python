#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUP="${1:?group result directory required}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PYTHONPATH="$ROOT/tools" "$PYTHON_BIN" "$ROOT/tools/validate_gate5_geometry.py" \
  --cases "$GROUP/cases" --out "$GROUP/QK_GATE5_GEOMETRY_PREFLIGHT.json"
echo "QK_GATE5_GEOMETRY_SUMMARY_PASS"
