#!/bin/bash
set -euo pipefail

: "${JCP9_COMMIT:?Set JCP9_COMMIT to the pinned GitHub commit in the one-line command}"

JCP9_BASE=${JCP9_BASE:-/project/pi_roohie_umass_edu/DSMC_Python_M3_QY}
JCP9_ROOT=${JCP9_ROOT:-$JCP9_BASE/JCP9_M12_CLOSEOUT}
JCP9_PREDICTION=${JCP9_PREDICTION:-$JCP9_BASE/JCP7_M12_EVALUATION/JCP7_M12_PREDICTION_LOCK.zip}
JCP9_REFERENCE=${JCP9_REFERENCE:-$JCP9_BASE/JCP8_M12_REFERENCE/JCP8_M12_REFERENCE.zip}
JCP9_SCORE=${JCP9_SCORE:-$JCP9_BASE/JCP8_M12_REFERENCE/JCP8_M12_SCORE.zip}
JCP9_RAW=https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/$JCP9_COMMIT

for path in "$JCP9_PREDICTION" "$JCP9_REFERENCE" "$JCP9_SCORE"; do
  test -s "$path" || { echo "MISSING_LOCKED_INPUT=$path" >&2; exit 2; }
done

mkdir -p "$JCP9_ROOT/code" "$JCP9_ROOT/logs"
curl -fsSL "$JCP9_RAW/scripts/jcp9_closeout_existing_m12.py" -o "$JCP9_ROOT/code/jcp9_closeout_existing_m12.py"
curl -fsSL "$JCP9_RAW/reference_data/mohammadzadeh_2012/jcp9_m12_closeout_protocol.json" -o "$JCP9_ROOT/code/jcp9_m12_closeout_protocol.json"
curl -fsSL "$JCP9_RAW/scripts/unity_jcp9_closeout.sbatch" -o "$JCP9_ROOT/code/unity_jcp9_closeout.sbatch"

JCP9_JOB_ID=$(sbatch --parsable \
  --chdir="$JCP9_ROOT/logs" \
  --export=ALL,JCP9_ROOT="$JCP9_ROOT",JCP9_PREDICTION="$JCP9_PREDICTION",JCP9_REFERENCE="$JCP9_REFERENCE",JCP9_SCORE="$JCP9_SCORE" \
  "$JCP9_ROOT/code/unity_jcp9_closeout.sbatch")

printf 'JCP9_CLOSEOUT_JOB_ID=%s\n' "$JCP9_JOB_ID"
printf 'MONITOR=squeue -j %s\n' "$JCP9_JOB_ID"
printf 'WHEN_COMPLETE_UPLOAD=%s/JCP9_M12_CLOSEOUT.zip %s/JCP9_M12_CLOSEOUT.zip.sha256\n' "$JCP9_ROOT" "$JCP9_ROOT"
