#!/bin/bash
set -euo pipefail
ROOT="${JFM_ROOT:-${SLURM_SUBMIT_DIR:?}}"
OUTPUT_ROOT="${JFM_OUTPUT_ROOT:-$ROOT/run_output}"
EXISTING_ROOT="${JFM_EXISTING_FIG5_ROOT:-/project/pi_roohie_umass_edu/JFM_revision_2026/JFM_HIGHSTAT_FIGURES_80M_S100K/run_output/vram48/runs}"
cd "$ROOT"
mamba run --no-capture-output -n dsmc-gpu python -u tools/summarize_single.py \
  --existing-input "$EXISTING_ROOT" \
  --new-input "$OUTPUT_ROOT/l40s/runs" \
  --output "$OUTPUT_ROOT/summary" \
  --case-table cases/final7.csv
