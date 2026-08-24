#!/usr/bin/env bash
set -Eeuo pipefail

JCP11_CODE_COMMIT=91061854df382f85a1817defdb38c9db19524848
JCP11_RAW="https://raw.githubusercontent.com/Ehsan-Roohi/DSMC_Python/${JCP11_CODE_COMMIT}"
JCP11_ROOT="${JCP_ROOT:-/project/pi_roohie_umass_edu/DSMC_Python_M3_QY}"
JCP11_WORK="${JCP11_OUTPUT:-${JCP11_ROOT}/JCP11_SUPPORT_VALIDATION}"
JCP11_CODE="${JCP11_WORK}/code"
JCP11_SOURCE_DIR="${JCP11_SOURCE_DIR:-/project/pi_roohie_umass_edu/Ab-initio-shock/ABINITIO_SHOCK_TESTS_v2/DS2V_BIRD_M10_FRESH_ONLY_20260722_234904/source}"
JCP11_DATA_ROOT="${JCP11_DATA_ROOT:-${JCP11_ROOT}/vision_guided_dsmc/mv11_ds2v_cylinder_runs}"
JCP11_MODEL_LOCK="${JCP11_MODEL_LOCK:-${JCP11_ROOT}/JCP6R_MODEL_LOCK/JCP6R_MODEL_LOCK.zip}"
JCP11_REFERENCE="${JCP11_REFERENCE:-${JCP11_ROOT}/JCP8_M12_REFERENCE/JCP8_M12_REFERENCE.zip}"
EXPECTED_DATA_SHA256=a13e82650ffa7a0303b0353ad385b198839c2c738df7cff98ce343806e736b96
EXPECTED_HEAT_BENCH_SHA256=2d94da3d86786afd1c497994cad935cfca1d188d9431bf16960fbc533e3f6c34
EXPECTED_REFERENCE_SHA256=340dd425239d3df48a056b618caf49b1af22348e384ae9ab3ae597c5ba587f12

trap 'RC=$?; echo "JCP11_BOOTSTRAP_FAILED rc=${RC} line=${LINENO} command=${BASH_COMMAND}" >&2; exit "${RC}"' ERR

if [[ -f "${JCP11_WORK}/JCP11_SUPPORT_VALIDATION.zip" && -f "${JCP11_WORK}/JCP11_SUPPORT_VALIDATION.zip.sha256" ]]; then
  (cd "${JCP11_WORK}" && sha256sum -c JCP11_SUPPORT_VALIDATION.zip.sha256)
  echo "JCP11_ALREADY_COMPLETE=1"
  echo "UPLOAD=${JCP11_WORK}/JCP11_SUPPORT_VALIDATION.zip ${JCP11_WORK}/JCP11_SUPPORT_VALIDATION.zip.sha256"
  exit 0
fi
if [[ -f "${JCP11_WORK}/LAST_JCP11.env" ]]; then
  # Never submit duplicate fresh trajectories.  A failed chain is repaired
  # explicitly after its accounting and logs are inspected.
  source "${JCP11_WORK}/LAST_JCP11.env"
  sacct -X -j "${JCP11_ARRAY_JOB_ID},${JCP11_LOCK_JOB_ID},${JCP11_SCORE_JOB_ID}" \
    --format=JobID%22,JobName%18,State%20,Elapsed,ExitCode || true
  echo "JCP11_EXISTING_CHAIN=1"
  echo "MONITOR=squeue -j ${JCP11_ARRAY_JOB_ID},${JCP11_LOCK_JOB_ID},${JCP11_SCORE_JOB_ID}"
  exit 0
fi

[[ -f "${JCP11_SOURCE_DIR}/Plasma_Calculations2.bird_m10_fresh.F90" ]] || { echo "MISSING_DS2V_SOURCE=1" >&2; exit 2; }
[[ -f "${JCP11_MODEL_LOCK}" ]] || { echo "MISSING_JCP6R_MODEL_LOCK=1" >&2; exit 2; }
[[ -f "${JCP11_REFERENCE}" ]] || { echo "MISSING_JCP8_REFERENCE=1" >&2; exit 2; }
[[ "$(sha256sum "${JCP11_REFERENCE}" | awk '{print $1}')" == "${EXPECTED_REFERENCE_SHA256}" ]] || { echo "JCP8_REFERENCE_CHECKSUM_MISMATCH=1" >&2; exit 2; }

JCP11_DATA=
while IFS= read -r CANDIDATE; do
  if [[ "$(sha256sum "${CANDIDATE}" | awk '{print $1}')" == "${EXPECTED_DATA_SHA256}" ]]; then
    JCP11_DATA="${CANDIDATE}"
    break
  fi
done < <(find "${JCP11_DATA_ROOT}" -type f -path '*/input/DS2VD.DAT' -print)
JCP11_HEAT_BENCH=
while IFS= read -r CANDIDATE; do
  if [[ "$(sha256sum "${CANDIDATE}" | awk '{print $1}')" == "${EXPECTED_HEAT_BENCH_SHA256}" ]]; then
    JCP11_HEAT_BENCH="${CANDIDATE}"
    break
  fi
done < <(find "${JCP11_DATA_ROOT}" -type f -path '*/input/HEAT-BENCH.TXT' -print)
[[ -n "${JCP11_DATA}" && -f "${JCP11_DATA}" ]] || { echo "MISSING_LOCKED_DS2V_DATA=1" >&2; exit 2; }
[[ -n "${JCP11_HEAT_BENCH}" && -f "${JCP11_HEAT_BENCH}" ]] || { echo "MISSING_LOCKED_HEAT_BENCH=1" >&2; exit 2; }

if [[ -z "${JCP11_MV17B_CAMPAIGN:-}" ]]; then
  while IFS= read -r CANDIDATE; do
    ROOT_CANDIDATE="${CANDIDATE%/cases/pair_01_observation/results/moments/MV11_MOMENTS_NOUT0100.DAT}"
    VALID=1
    for PAIR in 01 02 03 04 05 06; do
      for NOUT in 0100 0108 0116; do
        [[ -f "${ROOT_CANDIDATE}/cases/pair_${PAIR}_observation/results/moments/MV11_MOMENTS_NOUT${NOUT}.DAT" ]] || VALID=0
      done
    done
    if [[ "${VALID}" == 1 ]]; then
      JCP11_MV17B_CAMPAIGN="${ROOT_CANDIDATE}"
      break
    fi
  done < <(find "${JCP11_ROOT}" -type f -path '*/cases/pair_01_observation/results/moments/MV11_MOMENTS_NOUT0100.DAT' -print)
fi
[[ -n "${JCP11_MV17B_CAMPAIGN:-}" && -d "${JCP11_MV17B_CAMPAIGN}/cases" ]] || { echo "MISSING_MV17B_CAMPAIGN=1" >&2; exit 2; }

# Any pre-existing fresh unit would make the first-run chronology ambiguous.
if find "${JCP11_WORK}/units" -mindepth 1 -maxdepth 1 -type d -name 'seed_260829*' -print -quit 2>/dev/null | grep -q .; then
  echo "PREEXISTING_JCP11_FRESH_UNIT=1" >&2
  exit 2
fi

mkdir -p "${JCP11_CODE}/scripts" "${JCP11_CODE}/reference_data/mohammadzadeh_2012" \
  "${JCP11_WORK}/logs" "${JCP11_WORK}/units" "${JCP11_WORK}/cases"
FILES=(
  scripts/jcp11_support_validation.py
  scripts/prepare_jcp3_ds2v_m12.py
  scripts/unity_jcp11_m12_evaluation.sbatch
  scripts/unity_jcp11_collect_lock.sbatch
  scripts/unity_jcp11_score_pack.sbatch
  reference_data/mohammadzadeh_2012/jcp11_support_validation_protocol.json
)
for FILE in "${FILES[@]}"; do
  mkdir -p "${JCP11_CODE}/$(dirname "${FILE}")"
  curl --retry 3 -fsSL "${JCP11_RAW}/${FILE}" -o "${JCP11_CODE}/${FILE}"
done
python -m py_compile "${JCP11_CODE}/scripts/jcp11_support_validation.py" \
  "${JCP11_CODE}/scripts/prepare_jcp3_ds2v_m12.py"
bash -n "${JCP11_CODE}/scripts/unity_jcp11_m12_evaluation.sbatch" \
  "${JCP11_CODE}/scripts/unity_jcp11_collect_lock.sbatch" \
  "${JCP11_CODE}/scripts/unity_jcp11_score_pack.sbatch"

PROTOCOL="${JCP11_CODE}/reference_data/mohammadzadeh_2012/jcp11_support_validation_protocol.json"
python "${JCP11_CODE}/scripts/jcp11_support_validation.py" freeze-rule \
  --model-lock "${JCP11_MODEL_LOCK}" --protocol "${PROTOCOL}" --output "${JCP11_WORK}"
(cd "${JCP11_WORK}" && sha256sum -c JCP11_SUPPORT_RULE_LOCK.zip.sha256)
python "${JCP11_CODE}/scripts/jcp11_support_validation.py" specificity \
  --model-lock "${JCP11_MODEL_LOCK}" \
  --rule "${JCP11_WORK}/JCP11_SUPPORT_RULE_LOCK.json" \
  --campaign "${JCP11_MV17B_CAMPAIGN}" --output "${JCP11_WORK}"
(cd "${JCP11_WORK}" && sha256sum -c JCP11_M10_SPECIFICITY.zip.sha256)

python - "${JCP11_WORK}" "${JCP11_CODE_COMMIT}" "${JCP11_MODEL_LOCK}" "${JCP11_REFERENCE}" "${JCP11_MV17B_CAMPAIGN}" <<'PY'
import hashlib, json, pathlib, sys
work, commit, model, reference, campaign = map(pathlib.Path, sys.argv[1:])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
value = {
    "stage": "JCP11_pre_submission_lock",
    "status": "rule_and_specificity_locked_before_fresh_array_submission",
    "code_commit": str(commit),
    "model_lock_archive_sha256": sha(model),
    "reference_archive_sha256": sha(reference),
    "mv17b_campaign": str(campaign),
    "support_rule_archive_sha256": sha(work / "JCP11_SUPPORT_RULE_LOCK.zip"),
    "specificity_archive_sha256": sha(work / "JCP11_M10_SPECIFICITY.zip"),
    "fresh_M12_unit_directories_present": False,
}
(work / "JCP11_PRE_SUBMISSION_LOCK.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY

ARRAY_JOB_ID="$(sbatch --parsable \
  --output="${JCP11_WORK}/logs/j11-eval_%A_%a.out" \
  --error="${JCP11_WORK}/logs/j11-eval_%A_%a.err" \
  --export="ALL,JCP11_SOURCE_DIR=${JCP11_SOURCE_DIR},JCP11_DATA=${JCP11_DATA},JCP11_HEAT_BENCH=${JCP11_HEAT_BENCH},JCP11_WORK=${JCP11_WORK},JCP11_CODE=${JCP11_CODE},JCP11_RULE_LOCK=${JCP11_WORK}/JCP11_SUPPORT_RULE_LOCK.json" \
  "${JCP11_CODE}/scripts/unity_jcp11_m12_evaluation.sbatch")"
LOCK_JOB_ID="$(sbatch --parsable --dependency="afterok:${ARRAY_JOB_ID}" \
  --output="${JCP11_WORK}/logs/j11-lock_%j.out" \
  --error="${JCP11_WORK}/logs/j11-lock_%j.err" \
  --export="ALL,JCP11_WORK=${JCP11_WORK},JCP11_CODE=${JCP11_CODE},JCP11_MODEL_LOCK=${JCP11_MODEL_LOCK},JCP11_RULE_LOCK=${JCP11_WORK}/JCP11_SUPPORT_RULE_LOCK.json" \
  "${JCP11_CODE}/scripts/unity_jcp11_collect_lock.sbatch")"
SCORE_JOB_ID="$(sbatch --parsable --dependency="afterok:${LOCK_JOB_ID}" \
  --output="${JCP11_WORK}/logs/j11-score_%j.out" \
  --error="${JCP11_WORK}/logs/j11-score_%j.err" \
  --export="ALL,JCP11_WORK=${JCP11_WORK},JCP11_CODE=${JCP11_CODE},JCP11_REFERENCE=${JCP11_REFERENCE}" \
  "${JCP11_CODE}/scripts/unity_jcp11_score_pack.sbatch")"
printf 'JCP11_ARRAY_JOB_ID=%q\nJCP11_LOCK_JOB_ID=%q\nJCP11_SCORE_JOB_ID=%q\nJCP11_WORK=%q\nJCP11_CODE_COMMIT=%q\n' \
  "${ARRAY_JOB_ID}" "${LOCK_JOB_ID}" "${SCORE_JOB_ID}" "${JCP11_WORK}" "${JCP11_CODE_COMMIT}" \
  > "${JCP11_WORK}/LAST_JCP11.env"

echo "JCP11_SUBMITTED=1"
echo "JCP11_ARRAY_JOB_ID=${ARRAY_JOB_ID}"
echo "JCP11_LOCK_JOB_ID=${LOCK_JOB_ID}"
echo "JCP11_SCORE_JOB_ID=${SCORE_JOB_ID}"
echo "NEW_DSMC_TRAJECTORIES=4"
echo "NEW_REFERENCE_TRAJECTORIES=0"
echo "MONITOR=squeue -j ${ARRAY_JOB_ID},${LOCK_JOB_ID},${SCORE_JOB_ID}"
echo "WHEN_COMPLETE_UPLOAD=${JCP11_WORK}/JCP11_SUPPORT_VALIDATION.zip ${JCP11_WORK}/JCP11_SUPPORT_VALIDATION.zip.sha256"
