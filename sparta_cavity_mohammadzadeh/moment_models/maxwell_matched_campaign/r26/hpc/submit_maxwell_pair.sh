#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUNDLE_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
: "${R26_MAXWELL_SEED_KN005:?export path to accepted KnGu=0.05 N40 state}"
: "${R26_MAXWELL_SEED_KN020:?export path to accepted KnGu=0.20 N20 state}"
OUT_ROOT=${OUT_ROOT:-$BUNDLE_ROOT/results}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULT_ROOT="$OUT_ROOT/pure_maxwell_$STAMP"
mkdir -p "$RESULT_ROOT"

COMMON_EXPORT="ALL,R26_MAXWELL_BUNDLE_ROOT=$BUNDLE_ROOT,R26_MAXWELL_RESULT_ROOT=$RESULT_ROOT,R26_MAXWELL_SEED_KN005=$R26_MAXWELL_SEED_KN005,R26_MAXWELL_SEED_KN020=$R26_MAXWELL_SEED_KN020"
K005=$(sbatch --parsable --chdir="$SCRIPT_DIR" --export="$COMMON_EXPORT" "$SCRIPT_DIR/r26_maxwell_kn005.slurm")
K020=$(sbatch --parsable --chdir="$SCRIPT_DIR" --export="$COMMON_EXPORT" "$SCRIPT_DIR/r26_maxwell_kn020.slurm")
printf 'r26_maxwell_kn005_job=%s\nr26_maxwell_kn020_job=%s\nresult_root=%s\n' "${K005%%;*}" "${K020%%;*}" "$RESULT_ROOT"
