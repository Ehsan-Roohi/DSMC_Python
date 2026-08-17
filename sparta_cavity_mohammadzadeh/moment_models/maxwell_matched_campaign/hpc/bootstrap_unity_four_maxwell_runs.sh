#!/usr/bin/env bash
set -euo pipefail
BRANCH=${MAXWELL_MODELS_BRANCH:-agent/maxwell-matched-antifourier}
BASE=${UNITY_MODEL_BASE:-/project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK}
REPO="$BASE/DSMC_Python_maxwell_matched_models"
URL=https://github.com/Ehsan-Roohi/DSMC_Python.git
if [[ -d "$REPO/.git" ]]; then
  [[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]] || { echo "tracked changes in $REPO" >&2; exit 3; }
  git -C "$REPO" fetch origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
  git -C "$REPO" switch "$BRANCH"
  git -C "$REPO" pull --ff-only origin "$BRANCH"
elif [[ -e "$REPO" ]]; then
  echo "path exists and is not a Git clone: $REPO" >&2
  exit 3
else
  git clone --branch "$BRANCH" --single-branch "$URL" "$REPO"
fi
BUNDLE="$REPO/sparta_cavity_mohammadzadeh/moment_models/maxwell_matched_campaign"
bash "$BUNDLE/hpc/submit_four_maxwell_runs.sh"
