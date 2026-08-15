#!/usr/bin/env bash
set -euo pipefail

BRANCH="agent/dsmcfoam-kn01-production-hq"
REPO_URL="https://github.com/Ehsan-Roohi/DSMC_Python.git"
PROJECT_ROOT="/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK"
CHECKOUT="$PROJECT_ROOT/DSMC_Python_dsmcFoam_kn01_hq"
LAST_ENV="$PROJECT_ROOT/LAST_DSMCFOAM_KN01_HQ_JOB.env"

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
RUN_ROOT="$PROJECT_ROOT/dsmcfoam_hq_runs"
LOG_DIR="$MATCH_ROOT/logs_hq"
mkdir -p "$LOG_DIR" "$RUN_ROOT"

ARRAY_ID=$(sbatch --parsable \
    --chdir="$MATCH_ROOT" \
    --export="ALL,BASE_CASE_ROOT=$BASE_CASE_ROOT,OVERRIDE_ROOT=$OVERRIDE_ROOT,MATCH_ROOT=$MATCH_ROOT,RUN_ROOT=$RUN_ROOT,LAST_ENV=$LAST_ENV" \
    "$MATCH_ROOT/hpc/unity_dsmcfoam_kn01_hq.slurm")
ARRAY_ID="${ARRAY_ID%%;*}"
CAMPAIGN_ROOT="$RUN_ROOT/${ARRAY_ID}_kn01_hq"

tmp_env="${LAST_ENV}.tmp.$$"
{
    printf 'export HQ_ARRAY_ID=%q\n' "$ARRAY_ID"
    printf 'export HQ_JOB_IDS=%q\n' "${ARRAY_ID}_0,${ARRAY_ID}_1,${ARRAY_ID}_2"
    printf 'export HQ_CAMPAIGN_ROOT=%q\n' "$CAMPAIGN_ROOT"
    printf 'export HQ_LOG_GLOB=%q\n' "$LOG_DIR/dsmcfoam-kn01-hq-${ARRAY_ID}_*.out"
} > "$tmp_env"
mv "$tmp_env" "$LAST_ENV"

echo "Submitted dsmcFoam Kn=0.1 HQ ensemble: ARRAY_ID=$ARRAY_ID"
echo "Members: ${ARRAY_ID}_0 ${ARRAY_ID}_1 ${ARRAY_ID}_2"
echo "Environment file: $LAST_ENV"
echo "Monitor: source '$LAST_ENV' && squeue -j \"\$HQ_ARRAY_ID\" && sacct -X -j \"\$HQ_ARRAY_ID\" --format=JobID%20,State,ExitCode,Elapsed,NodeList%20"

