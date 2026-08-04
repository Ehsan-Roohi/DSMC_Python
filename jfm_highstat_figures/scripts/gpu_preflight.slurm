#!/bin/bash
set -uo pipefail

ROOT="${JFM_ROOT:-${SLURM_SUBMIT_DIR:?}}"
ROUTE="${JFM_ROUTE:?JFM_ROUTE is required}"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$ROOT/run_output}"
PREFLIGHT_DIR="$OUTPUT_ROOT/$ROUTE/preflight"
MODE_FILE="$OUTPUT_ROOT/$ROUTE/mode.env"
mkdir -p "$PREFLIGHT_DIR"
cd "$ROOT"

echo "route=$ROUTE host=$(hostname) job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv

if [[ "$ROUTE" == "low" ]]; then
  REQUIRE_FREE=9.0
else
  REQUIRE_FREE=20.0
fi

COMMON=(
  --kn 20 --rt 0.5 --particles 40000000
  --steps 4 --sample-start 2 --sample-every 1
  --time-blocks 2 --progress-every 1
  --require-free-gb "$REQUIRE_FREE"
)

set +e
mamba run -n dsmc-gpu python solver/JFM_hs_dsmc_quarter.py \
  "${COMMON[@]}" --seed 900000001 --output "$PREFLIGHT_DIR/hs40m" \
  >"$PREFLIGHT_DIR/hs40m.log" 2>&1
HS_STATUS=$?
mamba run -n dsmc-gpu python solver/JFM_bgk_shakhov_quarter.py \
  --mode SHAKHOV "${COMMON[@]}" --seed 900000007 \
  --output "$PREFLIGHT_DIR/shakhov40m" \
  >"$PREFLIGHT_DIR/shakhov40m.log" 2>&1
SHAKHOV_STATUS=$?
set -e

if [[ "$HS_STATUS" == 0 && "$SHAKHOV_STATUS" == 0 ]]; then
  printf 'JFM_LOW_MODE=40m\nJFM_PARTICLES=40000000\n' > "$MODE_FILE"
  echo "[OK] 40M HS and Shakhov memory preflight passed"
  exit 0
fi

echo "40M preflight status: HS=$HS_STATUS SHAKHOV=$SHAKHOV_STATUS" >&2
if [[ "$ROUTE" == "low" ]]; then
  printf 'JFM_LOW_MODE=20m_split\nJFM_PARTICLES=20000000\n' > "$MODE_FILE"
  echo "[FALLBACK] Low-VRAM production will use six independent 20M runs per case"
  exit 0
fi

echo "[FAIL] 40M does not fit the requested high-memory GPU" >&2
exit 3
