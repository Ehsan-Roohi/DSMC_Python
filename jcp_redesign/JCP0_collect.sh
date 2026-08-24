#!/bin/bash
# Collect the code and configuration needed to implement the JCP redesign.
# This does not launch DSMC and does not copy large result arrays.

set -euo pipefail

JCP_ROOT=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/vision_guided_dsmc
JCP_DS2V=/project/pi_roohie_umass_edu/Ab-initio-shock/ABINITIO_SHOCK_TESTS_v2/DS2V_BIRD_M10_FRESH_ONLY_20260722_234904/source/Plasma_Calculations2.bird_m10_fresh.F90
JCP_OUT=/project/pi_roohie_umass_edu/DSMC_Python_M3_QY/JCP0
JCP_STAGE=$(mktemp -d "$JCP_OUT.stage.XXXXXX")

test -d "$JCP_ROOT"
test -f "$JCP_DS2V"
mkdir -p "$JCP_OUT"

cd "$JCP_ROOT"
JCP_LIST="$JCP_STAGE/files.txt"
: > "$JCP_LIST"
for d in vgdsmc scripts tests reference_data mv11_ds2v_cylinder_runs; do
    if test -d "$d"; then
        find "$d" -type f \
            ! -path '*/results/*' \
            ! -path '*/__pycache__/*' \
            \( -name '*.py' -o -name '*.sh' -o -name '*.sbatch' \
               -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' \
               -o -name '*.toml' -o -name '*.env' -o -name '*.tsv' \
               -o -name '*.csv' -o -name '*.txt' -o -name '*.md' \
               -o -name '*.F90' -o -name '*.f90' -o -name 'Makefile' \
               -o -name 'requirements*' -o -name 'pyproject.toml' \) \
            -print >> "$JCP_LIST"
    fi
done
sort -u "$JCP_LIST" -o "$JCP_LIST"

while IFS= read -r f; do
    mkdir -p "$JCP_STAGE/repo/$(dirname "$f")"
    cp -p -- "$f" "$JCP_STAGE/repo/$f"
done < "$JCP_LIST"

mkdir -p "$JCP_STAGE/external"
cp -p -- "$JCP_DS2V" "$JCP_STAGE/external/DS2V_M10.F90"

if git -C "$JCP_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$JCP_ROOT" rev-parse HEAD > "$JCP_STAGE/git_head.txt"
    git -C "$JCP_ROOT" status --short > "$JCP_STAGE/git_status.txt"
    git -C "$JCP_ROOT" log -5 --oneline > "$JCP_STAGE/git_log.txt"
fi

find "$JCP_STAGE" -type f -printf '%P\t%s\n' | sort > "$JCP_STAGE/inventory.tsv"
JCP_ARCHIVE="$JCP_OUT/JCP0_src.tar.gz"
test ! -e "$JCP_ARCHIVE"
tar -czf "$JCP_ARCHIVE" -C "$JCP_STAGE" .
sha256sum "$JCP_ARCHIVE" > "$JCP_ARCHIVE.sha256"

printf 'UPLOAD THESE TWO FILES:\n%s\n%s\n' "$JCP_ARCHIVE" "$JCP_ARCHIVE.sha256"
