#!/bin/bash
set -uo pipefail

ROOT="${JFM_ROOT:-${SLURM_SUBMIT_DIR:?}}"
ROUTE="${JFM_ROUTE:?JFM_ROUTE is required}"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$ROOT/run_output}"
PREFLIGHT_DIR="$OUTPUT_ROOT/$ROUTE/preflight"
mkdir -p "$PREFLIGHT_DIR"
cd "$ROOT"

echo "route=$ROUTE host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv

case "$ROUTE" in
  vram48) REQUIRE_FREE=40.0 ;;
  vram80) REQUIRE_FREE=60.0 ;;
  *) echo "Unknown route: $ROUTE" >&2; exit 2 ;;
esac

COMMON=(
  --kn 20 --rt 0.5 --particles 80000000
  --steps 4 --sample-start 2 --sample-every 1
  --time-blocks 2 --progress-every 1
  --require-free-gb "$REQUIRE_FREE"
)

set +e
mamba run -n dsmc-gpu python solver/JFM_hs_dsmc_quarter.py \
  "${COMMON[@]}" --seed 900000001 --output "$PREFLIGHT_DIR/hs80m" \
  >"$PREFLIGHT_DIR/hs80m.log" 2>&1
HS_STATUS=$?
mamba run -n dsmc-gpu python solver/JFM_bgk_shakhov_quarter.py \
  --mode SHAKHOV "${COMMON[@]}" --seed 900000007 \
  --output "$PREFLIGHT_DIR/shakhov80m" \
  >"$PREFLIGHT_DIR/shakhov80m.log" 2>&1
SHAKHOV_STATUS=$?
set -e

if [[ "$HS_STATUS" == 0 && "$SHAKHOV_STATUS" == 0 ]]; then
  echo "[OK] 80M HS and Shakhov memory preflight passed"
  exit 0
fi

echo "80M preflight status: HS=$HS_STATUS SHAKHOV=$SHAKHOV_STATUS" >&2
echo "[FAIL] 80M does not fit the allocated GPU; production array was not released" >&2
exit 3
