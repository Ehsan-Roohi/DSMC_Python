#!/usr/bin/env bash
#SBATCH --job-name=dsmcfoam-kn01
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --constraint=avx512
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/dsmcfoam-kn01-%j.out
#SBATCH --error=logs/dsmcfoam-kn01-%j.err

set -euo pipefail

BASE_CASE_ROOT="${BASE_CASE_ROOT:?BASE_CASE_ROOT was not exported by run_unity.sh}"
OVERRIDE_ROOT="${OVERRIDE_ROOT:?OVERRIDE_ROOT was not exported by run_unity.sh}"
MATCH_ROOT="${MATCH_ROOT:?MATCH_ROOT was not exported by run_unity.sh}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT was not exported by run_unity.sh}"
LAST_ENV="${LAST_ENV:?LAST_ENV was not exported by run_unity.sh}"

source_openfoam_environment() {
    local root rc
    local -a roots=()
    [[ -n "${EBROOTOPENFOAM:-}" ]] && roots+=("$EBROOTOPENFOAM")
    [[ -n "${WM_PROJECT_DIR:-}" ]] && roots+=("$WM_PROJECT_DIR")
    for root in "${roots[@]}"; do
        for rc in "$root/etc/bashrc" "$root/OpenFOAM-v2406/etc/bashrc" "$root/OpenFOAM-v2006/etc/bashrc"; do
            [[ -r "$rc" ]] || continue
            set +u
            source "$rc" >/dev/null 2>&1 || true
            set -u
            if command -v dsmcFoam >/dev/null 2>&1; then
                export DSMCFOAM_BASHRC_USED="$rc"
                return 0
            fi
        done
    done
    return 1
}

load_openfoam() {
    if command -v dsmcFoam >/dev/null 2>&1; then
        return 0
    fi
    if [[ -n "${DSMCFOAM_BASHRC:-}" && -r "$DSMCFOAM_BASHRC" ]]; then
        set +u
        source "$DSMCFOAM_BASHRC"
        set -u
        if command -v dsmcFoam >/dev/null 2>&1; then
            export DSMCFOAM_BASHRC_USED="$DSMCFOAM_BASHRC"
            return 0
        fi
    fi
    if command -v module >/dev/null 2>&1; then
        module purge >/dev/null 2>&1 || true
        module load uri/main
        module load OpenFOAM/v2406-foss-2023a
        export DSMCFOAM_MODULE="OpenFOAM/v2406-foss-2023a"
        command -v dsmcFoam >/dev/null 2>&1 && return 0
        source_openfoam_environment && return 0
    fi
    local rc
    shopt -s nullglob
    for rc in \
        /project/pi_roohie_umass_edu/OpenFOAM/*/etc/bashrc \
        /project/pi_roohie_umass_edu/openfoam*/etc/bashrc \
        /project/pi_roohie_umass_edu/DSMC_CAVITY_BOOK/OpenFOAM/*/etc/bashrc \
        /work/pi_roohie_umass_edu/roohie_umass_edu/OpenFOAM/*/etc/bashrc \
        /opt/openfoam*/etc/bashrc; do
        set +u
        source "$rc" >/dev/null 2>&1 || true
        set -u
        if command -v dsmcFoam >/dev/null 2>&1; then
            export DSMCFOAM_BASHRC_USED="$rc"
            shopt -u nullglob
            return 0
        fi
    done
    shopt -u nullglob
    return 1
}

diagnose_openfoam_environment() {
    echo "===== OPENFOAM MODULE DIAGNOSTICS =====" >&2
    hostname >&2
    module list 2>&1 || true
    printf 'EBROOTOPENFOAM=%s\n' "${EBROOTOPENFOAM:-unset}" >&2
    printf 'WM_PROJECT_DIR=%s\n' "${WM_PROJECT_DIR:-unset}" >&2
    printf 'FOAM_APPBIN=%s\n' "${FOAM_APPBIN:-unset}" >&2
    printf 'FOAM_USER_APPBIN=%s\n' "${FOAM_USER_APPBIN:-unset}" >&2
}

if ! load_openfoam; then
    echo "ERROR: dsmcFoam was not found after loading the module and sourcing its etc/bashrc." >&2
    diagnose_openfoam_environment
    exit 127
fi

for command_name in blockMesh checkMesh dsmcInitialise decomposePar dsmcFoam reconstructPar mpirun; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "ERROR: required command not found: $command_name" >&2
        diagnose_openfoam_environment
        exit 127
    }
done

mkdir -p "$RUN_ROOT"
RUN_DIR="$RUN_ROOT/${SLURM_JOB_ID}_kn01"
if [[ -e "$RUN_DIR" ]]; then
    echo "ERROR: run directory already exists: $RUN_DIR" >&2
    exit 2
fi
cp -a "$BASE_CASE_ROOT" "$RUN_DIR"
cp -a "$OVERRIDE_ROOT"/. "$RUN_DIR"/
cd "$RUN_DIR"

{
    echo "job_id=$SLURM_JOB_ID"
    echo "host=$(hostname)"
    echo "started_utc=$(date -u +%FT%TZ)"
    echo "solver=dsmcFoam"
    echo "comparison_target=SPARTA_Kn0.1_job_62778322"
    echo "kn=0.1"
    echo "number_density_m-3=1.294383653016e25"
    echo "mesh=80x80x1"
    echo "target_particles_per_cell=32"
    echo "openfoam_module=${DSMCFOAM_MODULE:-user-local/bashrc}"
    echo "openfoam_bashrc=${DSMCFOAM_BASHRC_USED:-module-path}"
    echo "dsmcFoam=$(command -v dsmcFoam)"
    command -v foamVersion >/dev/null 2>&1 && foamVersion || true
} | tee RUN_METADATA.txt

CURRENT_STAGE="case-setup"
report_failure() {
    local rc=$?
    (( rc != 0 )) || return 0
    set +e
    echo "[stage:${CURRENT_STAGE}] FAILED exit_code=$rc" >&2
    {
        echo "finished_utc=$(date -u +%FT%TZ)"
        echo "status=FAILED"
        echo "failed_stage=$CURRENT_STAGE"
        echo "exit_code=$rc"
    } >> RUN_METADATA.txt
    if [[ -f "log.${CURRENT_STAGE}" ]]; then
        echo "===== tail log.${CURRENT_STAGE} =====" >&2
        tail -120 "log.${CURRENT_STAGE}" >&2
    fi
}
trap report_failure EXIT

run_stage() {
    local stage=$1
    shift
    CURRENT_STAGE="$stage"
    echo "[stage:${stage}] START $(date -u +%FT%TZ)"
    "$@" > "log.${stage}" 2>&1
    echo "[stage:${stage}] PASS $(date -u +%FT%TZ)"
}

verify_case_contract() {
    grep -Fq 'Ar      1.294383653016e25;' system/dsmcInitialiseDict
    grep -Fq 'nEquivalentParticles    0.7900290850925;' constant/dsmcProperties
    grep -Fq 'hex (0 1 2 3 4 5 6 7) (80 80 1)' system/blockMeshDict
    grep -Fq 'deltaT              5e-12;' system/controlDict
    grep -Fq 'endTime             1e-7;' system/controlDict
    echo 'KN01_CASE_CONTRACT_PASS'
}

verify_parallel_initial_state() {
    local proc field
    local -a required_fields=(
        boundaryT boundaryU dsmcRhoN fD iDof internalE linearKE
        momentum q rhoM rhoN
    )
    for proc in processor{0..15}; do
        [[ -d "$proc/0" ]] || {
            echo "ERROR: missing zero-time directory: $proc/0" >&2
            return 1
        }
        for field in "${required_fields[@]}"; do
            [[ -f "$proc/0/$field" ]] || {
                echo "ERROR: missing decomposed field: $proc/0/$field" >&2
                return 1
            }
        done
    done
}

run_stage caseContract verify_case_contract
run_stage blockMesh blockMesh
run_stage checkMesh checkMesh
run_stage dsmcInitialise dsmcInitialise
run_stage decomposePar decomposePar -force
run_stage initialState verify_parallel_initial_state
run_stage dsmcFoam mpirun -np "$SLURM_NTASKS" dsmcFoam -parallel
run_stage reconstructPar reconstructPar -latestTime

CURRENT_STAGE="finalize"
{
    echo "finished_utc=$(date -u +%FT%TZ)"
    echo "status=COMPLETED"
} | tee -a RUN_METADATA.txt

tmp_env="${LAST_ENV}.tmp.${SLURM_JOB_ID}"
{
    printf 'export JOB_ID=%q\n' "$SLURM_JOB_ID"
    printf 'export CASE_DIR=%q\n' "$RUN_DIR"
    printf 'export SLURM_LOG=%q\n' "$MATCH_ROOT/logs/dsmcfoam-kn01-${SLURM_JOB_ID}.out"
} > "$tmp_env"
mv "$tmp_env" "$LAST_ENV"
trap - EXIT

