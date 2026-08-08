#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${MV3_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${REPO_ROOT}/LAST_MOHAMMADZADEH_VISION_MV3_JOB.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing prior MV3 job environment: ${ENV_FILE}" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"
PRIOR_REFERENCE_JOB_ID="${REFERENCE_JOB_ID:?missing REFERENCE_JOB_ID in prior environment}"
M3_ROOT="${MV3_M3_ROOT:?missing MV3_M3_ROOT in prior environment}"
OUTPUT_ROOT="${MV3_OUTPUT_ROOT:?missing MV3_OUTPUT_ROOT in prior environment}"
VENV_DIR="${MV3_VENV_DIR:?missing MV3_VENV_DIR in prior environment}"

if [[ -e "${OUTPUT_ROOT}/summary.json" ]]; then
  echo "Refusing to overwrite completed MV3 output: ${OUTPUT_ROOT}" >&2
  exit 2
fi
if find "${OUTPUT_ROOT}/tasks" -name summary.json -print -quit 2>/dev/null | grep -q .; then
  echo "Refusing recovery because completed model-task summaries already exist." >&2
  exit 2
fi

cd "${REPO_ROOT}"
source "${VENV_DIR}/bin/activate"
python -m pip install -e . --no-deps
python -m vgdsmc.mohammadzadeh_mv3_reference --verify-lock-only >/dev/null

export MV3_M3_ROOT="${M3_ROOT}"
export MV3_OUTPUT_ROOT="${OUTPUT_ROOT}"
python - <<'PY'
import os
from pathlib import Path

from vgdsmc.mohammadzadeh_mv3_reference import new_reference_tasks
from vgdsmc.mohammadzadeh_vision_mv3 import _verify_source

m3_root = Path(os.environ["MV3_M3_ROOT"])
output_root = Path(os.environ["MV3_OUTPUT_ROOT"])
for seed in range(91901, 91909):
    _verify_source(m3_root / f"seed_{seed}", "complete_M3_qy_precision_seed")
for condition_id, seed in new_reference_tasks():
    _verify_source(
        output_root / "references" / condition_id / f"seed_{seed}",
        "complete_MV3_reference_seed",
    )
print("Verified 8 reused M3 seeds and 12 completed MV3 references.")
PY

EXPORTS="ALL,MV3_REPO_ROOT=${REPO_ROOT},MV3_M3_ROOT=${M3_ROOT},MV3_OUTPUT_ROOT=${OUTPUT_ROOT},MV3_VENV_DIR=${VENV_DIR}"
MODEL_JOB_ID="$(sbatch --parsable --export="${EXPORTS}" scripts/unity_mohammadzadeh_vision_mv3_task.sbatch)"
POST_JOB_ID="$(sbatch --parsable --dependency="afterok:${MODEL_JOB_ID}" --export="${EXPORTS}" scripts/unity_mohammadzadeh_vision_mv3_post.sbatch)"

printf 'REFERENCE_JOB_ID=%q\nMODEL_JOB_ID=%q\nPOST_JOB_ID=%q\nMV3_REPO_ROOT=%q\nMV3_M3_ROOT=%q\nMV3_OUTPUT_ROOT=%q\nMV3_VENV_DIR=%q\n' \
  "${PRIOR_REFERENCE_JOB_ID}" "${MODEL_JOB_ID}" "${POST_JOB_ID}" "${REPO_ROOT}" "${M3_ROOT}" "${OUTPUT_ROOT}" "${VENV_DIR}" > "${ENV_FILE}"

echo "Reused completed MV3 references: ${PRIOR_REFERENCE_JOB_ID}"
echo "Submitted repaired MV3 model benchmark: ${MODEL_JOB_ID} (16 tasks, at most 4 concurrent)"
echo "Submitted repaired MV3 postprocessor: ${POST_JOB_ID}"
echo "Saved: ${ENV_FILE}"
echo "Monitor: squeue -j ${MODEL_JOB_ID},${POST_JOB_ID}"
