#!/usr/bin/env bash
set -euo pipefail

REPOSITORY="Ehsan-Roohi/DSMC_Python"
RESTART_REF="f97b3d6a8e97e2bc41ad1444f8d2c32e9e5b152f"
ENDPOINT_REF="65beda519fb4c40b9d9367c3108265313974cc48"
PROJECT_ROOT="/project/pi_roohie_umass_edu/JFM_revision_2026"
TARGET="$PROJECT_ROOT/JFM_RT05_KN005_HS_THREE_SEED"
OUTPUT_ROOT="$TARGET/run_output"

for command_name in curl tar sbatch python sha256sum mamba; do
    command -v "$command_name" >/dev/null || {
        echo "Missing required command: $command_name" >&2
        exit 2
    }
done

mkdir -p "$PROJECT_ROOT" "$TARGET"
temporary_directory="$(mktemp -d)"
trap 'rm -rf "$temporary_directory"' EXIT
mkdir -p "$temporary_directory/restart" "$temporary_directory/endpoint"

curl -fsSL \
    "https://codeload.github.com/${REPOSITORY}/tar.gz/${RESTART_REF}" \
    -o "$temporary_directory/restart.tar.gz"
curl -fsSL \
    "https://codeload.github.com/${REPOSITORY}/tar.gz/${ENDPOINT_REF}" \
    -o "$temporary_directory/endpoint.tar.gz"
tar -xzf "$temporary_directory/restart.tar.gz" \
    -C "$temporary_directory/restart"
tar -xzf "$temporary_directory/endpoint.tar.gz" \
    -C "$temporary_directory/endpoint"

restart_source="$(find "$temporary_directory/restart" -type d \
    -path '*/jfm_single_seed_restartable_fast' -print -quit)"
endpoint_source="$(find "$temporary_directory/endpoint" -type d \
    -path '*/jfm_rt05_kn001_heavy_rerun' -print -quit)"
[[ -n "$restart_source" && -n "$endpoint_source" ]] || {
    echo "Pinned JFM packages were not found" >&2
    exit 3
}

cp -a "$restart_source/." "$TARGET/"
cp "$endpoint_source/tools/summarize_ensembles.py" \
    "$TARGET/tools/summarize_ensembles.py"
cp "$endpoint_source/scripts/summarize_kn001.slurm" \
    "$TARGET/scripts/summarize_kn001.slurm"
cd "$TARGET"

[[ "$(sha256sum solver/JFM_hs_dsmc_quarter.py | awk '{print $1}')" == \
   "2c9e2f5119802b123f0335664a564085cfe77682ae4f4af4138dc31f5876b166" ]]

awk -F, 'BEGIN {
        OFS=",";
        print "case_id,model,kn,rt,seed,figure,source"
    }
    NR>1 && $2=="HS" {
        print sprintf("K005_%02d", count++), "HS", "0.05", "0.5", \
              $5, "Figure8b", "Din_request"
    }' "$endpoint_source/cases/kn001_heavy.csv" > cases/kn005_hs.csv

sed \
    -e 's/particles=80000000 steps=1500000 checkpoint=1000000 sample_window=100000:1500000:2 time_blocks=14/particles=22000000 steps=1000000 restart=1000000 sample_window=400000:1000000:2 time_blocks=3/' \
    -e 's/--kn "$KN" --rt "$RT" --particles 80000000/--kn "$KN" --rt "$RT" --particles 22000000/' \
    -e 's/--steps 1500000 --sample-start 100000/--steps 1000000 --sample-start 400000/' \
    -e 's/--time-blocks 14 --checkpoint-steps 1000000/--time-blocks 3/' \
    -e 's/--require-free-gb 40\.0/--require-free-gb 8.5/' \
    scripts/run_checkpoint_fast.slurm > scripts/run_kn005_segment1.slurm

sed \
    -e 's/SOURCE_STEPS="${JFM_CONTINUE_SOURCE_STEPS:-1500000}"/SOURCE_STEPS="${JFM_CONTINUE_SOURCE_STEPS:-1000000}"/' \
    -e 's/particles=80000000 sample_start=100000/particles=22000000 sample_start=400000/' \
    -e 's/--kn "$KN" --rt "$RT" --particles 80000000/--kn "$KN" --rt "$RT" --particles 22000000/' \
    -e 's/--steps "$TARGET_STEPS" --sample-start 100000/--steps "$TARGET_STEPS" --sample-start 400000/' \
    -e 's/--require-free-gb 40\.0/--require-free-gb 8.5/' \
    scripts/run_continuation.slurm > scripts/run_kn005_segment2.slurm

sed \
    -e 's/jfm-k001-sum/jfm-k005-sum/' \
    -e 's#"$OUTPUT_ROOT/kn001_heavy"#"$OUTPUT_ROOT/continuations/step2000000/runs"#' \
    -e 's/summary_kn001_heavy/summary_kn005_hs/g' \
    -e 's/cases\/kn001_heavy\.csv/cases\/kn005_hs.csv/g' \
    -e 's/Kn=0\.01/Kn=0.05/g' \
    scripts/summarize_kn001.slurm > scripts/summarize_kn005_hs.slurm

python -m py_compile \
    solver/JFM_hs_dsmc_quarter.py \
    tools/summarize_ensembles.py
bash -n scripts/run_kn005_segment1.slurm
bash -n scripts/run_kn005_segment2.slurm
bash -n scripts/summarize_kn005_hs.slurm

[[ "$(awk 'END{print NR-1}' cases/kn005_hs.csv)" == "3" ]]
[[ "$(awk -F, 'NR>1 {print $2}' cases/kn005_hs.csv | sort -u)" == "HS" ]]
[[ "$(awk -F, 'NR>1 {print $3}' cases/kn005_hs.csv | sort -u)" == "0.05" ]]
[[ "$(awk -F, 'NR>1 {print $4}' cases/kn005_hs.csv | sort -u)" == "0.5" ]]
[[ "$(awk -F, 'NR>1 {print $5}' cases/kn005_hs.csv | sort -nu | tr '\n' ' ')" == \
   "42 271828 314159 " ]]

mamba run -n dsmc-gpu python -c \
    'import cupy, numba, numpy, matplotlib; print("dsmc-gpu imports OK")'

mkdir -p \
    "$OUTPUT_ROOT/vram48/runs" \
    "$OUTPUT_ROOT/continuations/step2000000/runs" \
    "$OUTPUT_ROOT/summary_kn005_hs" \
    "$TARGET/slurm"
case_table="$TARGET/cases/kn005_hs.csv"
common_export="ALL,JFM_ROOT=$TARGET,JFM_OUTPUT_ROOT=$OUTPUT_ROOT,JFM_CASE_TABLE=$case_table"

segment1_job="$(sbatch --parsable \
    --job-name=jfm-k005-s1 \
    --partition=gpu --gpus=1 --constraint=2080ti \
    --cpus-per-task=4 --mem=32G --time=168:00:00 --array=0-2%3 \
    --output="$TARGET/slurm/jfm-k005-s1-%A_%a.out" \
    --error="$TARGET/slurm/jfm-k005-s1-%A_%a.err" \
    --export="$common_export" \
    scripts/run_kn005_segment1.slurm)"
echo "SEGMENT1_JOB=$segment1_job"

segment2_export="$common_export,JFM_CONTINUE_SOURCE_STEPS=1000000,JFM_CONTINUE_TARGET_STEPS=2000000,JFM_CONTINUE_TIME_BLOCKS=8"
segment2_job="$(sbatch --parsable \
    --dependency="afterok:$segment1_job" --kill-on-invalid-dep=yes \
    --job-name=jfm-k005-s2 \
    --partition=gpu --gpus=1 --constraint=2080ti \
    --cpus-per-task=4 --mem=32G --time=168:00:00 --array=0-2%3 \
    --output="$TARGET/slurm/jfm-k005-s2-%A_%a.out" \
    --error="$TARGET/slurm/jfm-k005-s2-%A_%a.err" \
    --export="$segment2_export" \
    scripts/run_kn005_segment2.slurm)"
echo "SEGMENT2_JOB=$segment2_job"

summary_job="$(sbatch --parsable \
    --dependency="afterok:$segment2_job" --kill-on-invalid-dep=yes \
    --job-name=jfm-k005-sum \
    --partition=cpu --cpus-per-task=4 --mem=32G --time=08:00:00 \
    --output="$TARGET/slurm/jfm-k005-summary-%j.out" \
    --error="$TARGET/slurm/jfm-k005-summary-%j.err" \
    --export="$common_export" \
    scripts/summarize_kn005_hs.slurm)"

printf 'SEGMENT1_JOB=%q\nSEGMENT2_JOB=%q\nSUMMARY_JOB=%q\nOUTPUT_ROOT=%q\n' \
    "$segment1_job" "$segment2_job" "$summary_job" "$OUTPUT_ROOT" \
    > LAST_KN005_SUBMISSION.env

echo "SUMMARY_JOB=$summary_job"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "Monitor: squeue -j $segment1_job,$segment2_job,$summary_job"
echo "[OK] HS Kn=0.05, RT=0.5, three-seed restartable Fig. 8(b) ensemble submitted"
