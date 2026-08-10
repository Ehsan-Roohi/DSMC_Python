#!/usr/bin/env bash
#SBATCH --job-name=qk_g5_phys
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/project/pi_roohie_umass_edu/Combustion/QK_GATE5_COUPLED/runs/physics-prescreen-%j.out
#SBATCH --error=/project/pi_roohie_umass_edu/Combustion/QK_GATE5_COUPLED/runs/physics-prescreen-%j.err
set -euo pipefail

BASE=${GATE5_BASE:-/project/pi_roohie_umass_edu/Combustion/QK_GATE5_COUPLED}
: "${GATE5_RESULT_DIR:?GATE5_RESULT_DIR is required}"
: "${GATE5_ANALYZER:?GATE5_ANALYZER is required}"
: "${GATE5_ANALYZER_SHA256:?GATE5_ANALYZER_SHA256 is required}"

if [[ ! -d "$GATE5_RESULT_DIR" ]]; then
  echo "ERROR: result directory does not exist: $GATE5_RESULT_DIR" >&2
  exit 2
fi
if [[ ! -f "$GATE5_ANALYZER" ]]; then
  echo "ERROR: analyzer does not exist: $GATE5_ANALYZER" >&2
  exit 2
fi
printf '%s  %s\n' "$GATE5_ANALYZER_SHA256" "$GATE5_ANALYZER" | sha256sum -c -

PYTHON_BIN=${GATE5_PYTHON_BIN:-}
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in \
    /work/pi_roohie_umass_edu/roohie_umass_edu/.conda/envs/dsmc-gpu/bin/python \
    "$BASE/.venvs/gate5-physics-cantera-3.1/bin/python" \
    python3; do
    if command -v "$candidate" >/dev/null 2>&1 || [[ -x "$candidate" ]]; then
      if "$candidate" -c 'import cantera; assert tuple(map(int, cantera.__version__.split(".")[:2])) >= (3, 0)' >/dev/null 2>&1; then
        PYTHON_BIN=$candidate
        break
      fi
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  BOOTSTRAP_PYTHON=$(command -v python3 || true)
  if [[ -z "$BOOTSTRAP_PYTHON" ]]; then
    echo "ERROR: python3 is unavailable" >&2
    exit 3
  fi
  VENV="$BASE/.venvs/gate5-physics-cantera-3.1"
  mkdir -p "$BASE/.venvs"
  if [[ ! -x "$VENV/bin/python" ]]; then
    "$BOOTSTRAP_PYTHON" -m venv "$VENV"
  fi
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  "$VENV/bin/python" -m pip install --only-binary=:all: 'cantera>=3.0,<3.2'
  PYTHON_BIN="$VENV/bin/python"
fi

"$PYTHON_BIN" -c 'import cantera; print("Cantera", cantera.__version__)'
"$PYTHON_BIN" "$GATE5_ANALYZER" --self-test

OUT="$BASE/runs/physics_prescreen_${SLURM_JOB_ID}"
mkdir -p "$OUT"
cp "$GATE5_ANALYZER" "$OUT/gate5_physics_prescreen.py"
printf '%s\n' \
  "JOB_ID=$SLURM_JOB_ID" \
  "SOURCE_RESULT_DIR=$GATE5_RESULT_DIR" \
  "ANALYZER_SHA256=$GATE5_ANALYZER_SHA256" \
  "PYTHON_BIN=$PYTHON_BIN" \
  "STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$OUT/RUN_PROVENANCE.txt"

"$PYTHON_BIN" "$GATE5_ANALYZER" \
  --result-dir "$GATE5_RESULT_DIR" \
  --out-dir "$OUT" \
  --ignition-horizon-s 0.02 \
  2>&1 | tee "$OUT/unity_gate5_physics_prescreen.log"

printf '%s\n' "COMPLETED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/RUN_PROVENANCE.txt"
echo "UNITY_QK_GATE5_PHYSICS_PRESCREEN_PASS"
echo "RESULT_DIR=$OUT"
