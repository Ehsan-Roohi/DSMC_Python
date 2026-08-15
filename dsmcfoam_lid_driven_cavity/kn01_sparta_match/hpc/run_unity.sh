#!/usr/bin/env bash
set -euo pipefail

BRANCH="agent/dsmcfoam-kn01-sparta-match"
REPO_URL="https://github.com/Ehsan-Roohi/DSMC_Python.git"
PROJECT_ROOT="/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK"
CHECKOUT="$PROJECT_ROOT/DSMC_Python_dsmcFoam_kn01"
LAST_ENV="$PROJECT_ROOT/LAST_DSMCFOAM_KN01_JOB.env"

if [[ -e "$CHECKOUT" && ! -d "$CHECKOUT/.git" ]]; then
    CHECKOUT="${CHECKOUT}_$(date +%Y%m%d_%H%M%S)"
fi

if [[ ! -d "$CHECKOUT/.git" ]]; then
    git clone --single-branch --branch "$BRANCH" "$REPO_URL" "$CHECKOUT"
else
    if [[ -n "$(git -C "$CHECKOUT" status --porcelain)" ]]; then
        CHECKOUT="${CHECKOUT}_$(date +%Y%m%d_%H%M%S)"
        git clone --single-branch --branch "$BRANCH" "$REPO_URL" "$CHECKOUT"
    else
        git -C "$CHECKOUT" fetch origin "$BRANCH"
        git -C "$CHECKOUT" switch -C "$BRANCH" FETCH_HEAD
    fi
fi

PACKAGE_ROOT="$CHECKOUT/dsmcfoam_lid_driven_cavity"
MATCH_ROOT="$PACKAGE_ROOT/kn01_sparta_match"
BASE_CASE_ROOT="$PACKAGE_ROOT/case"
OVERRIDE_ROOT="$MATCH_ROOT/case_overrides"
RUN_ROOT="$PROJECT_ROOT/dsmcfoam_runs"
mkdir -p "$MATCH_ROOT/logs" "$RUN_ROOT"

JOB_ID=$(sbatch --parsable \
    --chdir="$MATCH_ROOT" \
    --export="ALL,BASE_CASE_ROOT=$BASE_CASE_ROOT,OVERRIDE_ROOT=$OVERRIDE_ROOT,MATCH_ROOT=$MATCH_ROOT,RUN_ROOT=$RUN_ROOT,LAST_ENV=$LAST_ENV" \
    "$MATCH_ROOT/hpc/unity_dsmcfoam_kn01.slurm")
JOB_ID="${JOB_ID%%;*}"
SLURM_LOG="$MATCH_ROOT/logs/dsmcfoam-kn01-${JOB_ID}.out"

tmp_env="${LAST_ENV}.tmp.$$"
{
    printf 'export JOB_ID=%q\n' "$JOB_ID"
    printf 'export CASE_DIR=%q\n' "$RUN_ROOT/${JOB_ID}_kn01"
    printf 'export SLURM_LOG=%q\n' "$SLURM_LOG"
} > "$tmp_env"
mv "$tmp_env" "$LAST_ENV"

echo "Submitted OpenFOAM dsmcFoam Kn=0.1 cavity: JOB_ID=$JOB_ID"
echo "Environment file: $LAST_ENV"
echo "Monitor: source $LAST_ENV && squeue -j \"\$JOB_ID\" && tail -f \"\$SLURM_LOG\""

