#!/usr/bin/env bash
#SBATCH --job-name=moh_mv8_fig
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
: "${MV8_REPO_ROOT:?MV8_REPO_ROOT is required}"
: "${MV8_OUTPUT_ROOT:?MV8_OUTPUT_ROOT is required}"
: "${MV8_VENV_DIR:?MV8_VENV_DIR is required}"
: "${MV8_AUDIT_FIGURE_OUTPUT:?MV8_AUDIT_FIGURE_OUTPUT is required}"

cd "${MV8_REPO_ROOT}"
source "${MV8_VENV_DIR}/bin/activate"
python scripts/make_mohammadzadeh_mv8_audit_figures.py \
  --mv8-root "${MV8_OUTPUT_ROOT}" \
  --output "${MV8_AUDIT_FIGURE_OUTPUT}" \
  --repo-root "${MV8_REPO_ROOT}"

echo "MV8_AUDIT_FIGURE_OUTPUT=${MV8_AUDIT_FIGURE_OUTPUT}"
echo "MV8_AUDIT_FIGURE_ARCHIVE=${MV8_AUDIT_FIGURE_OUTPUT}/MOHAMMADZADEH_MV8_AUDIT_PHYSICAL_FIGURES.tar.gz"

