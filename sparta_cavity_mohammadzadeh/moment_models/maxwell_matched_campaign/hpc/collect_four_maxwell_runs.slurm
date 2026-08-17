#!/usr/bin/env bash
#SBATCH --job-name=mm4-pack
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-mm4-pack-%j.out
#SBATCH --error=slurm-mm4-pack-%j.err

set -euo pipefail
: "${MAXWELL_MODEL_BUNDLE_ROOT:?required}"
: "${MAXWELL_MODEL_RESULT_ROOT:?required}"
: "${MAXWELL_R26_RESULT_ROOT:?required}"
: "${MAXWELL_MODEL_DELIVERY_ROOT:?required}"
: "${MAXWELL_R13_JOB_ID:?required}"
: "${MAXWELL_R26_K005_JOB_ID:?required}"
: "${MAXWELL_R26_K020_JOB_ID:?required}"

mkdir -p "$MAXWELL_MODEL_DELIVERY_ROOT"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$MAXWELL_MODEL_DELIVERY_ROOT/R13_R26_MAXWELL_FOUR_RUNS_${MAXWELL_R13_JOB_ID}_${STAMP}_TO_ANALYZE.zip"
STAGE=$(mktemp -d /tmp/maxwell-four-send.XXXXXX)
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/results/r13" "$STAGE/results/r26" "$STAGE/provenance"

cp -a "$MAXWELL_MODEL_RESULT_ROOT/r13/." "$STAGE/results/r13/" 2>/dev/null || true
cp -a "$MAXWELL_R26_RESULT_ROOT/." "$STAGE/results/r26/" 2>/dev/null || true
cp -a "$MAXWELL_MODEL_BUNDLE_ROOT/r13/source" "$MAXWELL_MODEL_BUNDLE_ROOT/r13/tests" "$STAGE/provenance/"
cp -a "$MAXWELL_MODEL_BUNDLE_ROOT/r26/source" "$MAXWELL_MODEL_BUNDLE_ROOT/r26/tests" "$STAGE/provenance/"

for marker in \
  "$STAGE/results/r13/kn005_N60/VALIDATION.txt" \
  "$STAGE/results/r13/kn020_N60/VALIDATION.txt" \
  "$STAGE/results/r26/kn005_N40/FILES.sha256" \
  "$STAGE/results/r26/kn020_N20/FILES.sha256"; do
  [[ -s "$marker" ]] || echo "MISSING_OR_FAILED=$marker" >> "$STAGE/provenance/collection_status.txt"
done
printf 'r13_job=%s\nr26_kn005_job=%s\nr26_kn020_job=%s\n' \
  "$MAXWELL_R13_JOB_ID" "$MAXWELL_R26_K005_JOB_ID" "$MAXWELL_R26_K020_JOB_ID" \
  > "$STAGE/provenance/job_ids.txt"
(cd "$STAGE" && find . -type f ! -name FILES.sha256 -print0 | sort -z | xargs -0 sha256sum > FILES.sha256)
(cd "$STAGE" && zip -q -r -9 "$OUT" .)
unzip -tq "$OUT"
(cd "$(dirname "$OUT")" && sha256sum "$(basename "$OUT")" > "$(basename "$OUT").sha256.txt")
printf 'FINAL_ZIP=%s\nFINAL_SHA=%s.sha256.txt\n' "$OUT" "$OUT"
