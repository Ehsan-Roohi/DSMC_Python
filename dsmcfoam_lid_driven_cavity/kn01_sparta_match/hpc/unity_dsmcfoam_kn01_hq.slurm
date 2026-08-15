#!/usr/bin/env bash
#SBATCH --job-name=dsmc-kn01-hq
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --constraint=avx512
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs_hq/dsmcfoam-kn01-hq-%A_%a.out
#SBATCH --error=logs_hq/dsmcfoam-kn01-hq-%A_%a.err

set -euo pipefail

BASE_CASE_ROOT="${BASE_CASE_ROOT:?BASE_CASE_ROOT was not exported}"
OVERRIDE_ROOT="${OVERRIDE_ROOT:?OVERRIDE_ROOT was not exported}"
MATCH_ROOT="${MATCH_ROOT:?MATCH_ROOT was not exported}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT was not exported}"
LAST_ENV="${LAST_ENV:?LAST_ENV was not exported}"

# OpenFOAM-v2406 seeds the DSMC random generator from the MPI rank.  Using
# three different decompositions gives reproducible, decorrelated trajectories
# without changing the physical case.
case "$SLURM_ARRAY_TASK_ID" in
    0) MPI_NTASKS=16; PROC_X=4 ;;
    1) MPI_NTASKS=20; PROC_X=5 ;;
    2) MPI_NTASKS=24; PROC_X=6 ;;
    *) echo "ERROR: unsupported array member $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
PROC_Y=4

source_openfoam_environment() {
    local root rc
    local -a roots=()
    [[ -n "${EBROOTOPENFOAM:-}" ]] && roots+=("$EBROOTOPENFOAM")
    [[ -n "${WM_PROJECT_DIR:-}" ]] && roots+=("$WM_PROJECT_DIR")
    for root in "${roots[@]}"; do
        for rc in "$root/etc/bashrc" "$root/OpenFOAM-v2406/etc/bashrc"; do
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
    command -v dsmcFoam >/dev/null 2>&1 && return 0
    if [[ -n "${DSMCFOAM_BASHRC:-}" && -r "$DSMCFOAM_BASHRC" ]]; then
        set +u
        source "$DSMCFOAM_BASHRC"
        set -u
        command -v dsmcFoam >/dev/null 2>&1 && {
            export DSMCFOAM_BASHRC_USED="$DSMCFOAM_BASHRC"
            return 0
        }
    fi
    if command -v module >/dev/null 2>&1; then
        module purge >/dev/null 2>&1 || true
        module load uri/main
        module load OpenFOAM/v2406-foss-2023a
        export DSMCFOAM_MODULE="OpenFOAM/v2406-foss-2023a"
        command -v dsmcFoam >/dev/null 2>&1 && return 0
        source_openfoam_environment && return 0
    fi
    return 1
}

diagnose_openfoam_environment() {
    echo "===== OPENFOAM MODULE DIAGNOSTICS =====" >&2
    hostname >&2
    module list 2>&1 || true
    printf 'EBROOTOPENFOAM=%s\n' "${EBROOTOPENFOAM:-unset}" >&2
    printf 'WM_PROJECT_DIR=%s\n' "${WM_PROJECT_DIR:-unset}" >&2
    printf 'FOAM_APPBIN=%s\n' "${FOAM_APPBIN:-unset}" >&2
}

if ! load_openfoam; then
    echo "ERROR: dsmcFoam was not found." >&2
    diagnose_openfoam_environment
    exit 127
fi

for command_name in blockMesh checkMesh dsmcInitialise decomposePar dsmcFoam reconstructPar mpirun; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "ERROR: required command not found: $command_name" >&2
        exit 127
    }
done

CAMPAIGN_ROOT="$RUN_ROOT/${SLURM_ARRAY_JOB_ID}_kn01_hq"
RUN_DIR="$CAMPAIGN_ROOT/member_${SLURM_ARRAY_TASK_ID}_${MPI_NTASKS}r"
mkdir -p "$CAMPAIGN_ROOT"
if [[ -e "$RUN_DIR" ]]; then
    echo "ERROR: run directory already exists: $RUN_DIR" >&2
    exit 2
fi
cp -a "$BASE_CASE_ROOT" "$RUN_DIR"
cp -a "$OVERRIDE_ROOT"/. "$RUN_DIR"/
cd "$RUN_DIR"

# Four times more particles per cell: 32 -> 128.
sed -i 's/nEquivalentParticles    0.7900290850925;/nEquivalentParticles    0.197507271273125;/' constant/dsmcProperties

# 200,000 steps total; discard 40,000 and average 160,000 steps.
sed -i 's/endTime             1e-7;/endTime             1e-6;/' system/controlDict
sed -i 's/writeInterval       2000;/writeInterval       20000;/' system/controlDict
sed -i 's/purgeWrite          0;/purgeWrite          2;/' system/controlDict
sed -i 's/timeStart       2e-8;/timeStart       2e-7;/' system/controlDict

# Match the simple decomposition to this ensemble member.
sed -i -E "s/numberOfSubdomains[[:space:]]+[0-9]+;/numberOfSubdomains ${MPI_NTASKS};/" system/decomposeParDict
sed -i -E "s/n[[:space:]]+\([^)]*\);/n       (${PROC_X} ${PROC_Y} 1);/" system/decomposeParDict

{
    echo "array_job_id=$SLURM_ARRAY_JOB_ID"
    echo "array_task_id=$SLURM_ARRAY_TASK_ID"
    echo "host=$(hostname)"
    echo "started_utc=$(date -u +%FT%TZ)"
    echo "solver=dsmcFoam"
    echo "production_level=HQ_ENSEMBLE"
    echo "comparison_target=SPARTA_Kn0.1_job_62778322"
    echo "kn=0.1"
    echo "number_density_m-3=1.294383653016e25"
    echo "mesh=80x80x1"
    echo "target_particles_per_cell=128"
    echo "estimated_simulator_particles=819200"
    echo "delta_t_s=5e-12"
    echo "total_steps=200000"
    echo "warmup_steps=40000"
    echo "averaging_steps=160000"
    echo "mpi_ranks=$MPI_NTASKS"
    echo "ensemble_decorrelation=mpi_decomposition_${MPI_NTASKS}r"
    echo "openfoam_module=${DSMCFOAM_MODULE:-user-local/bashrc}"
    echo "openfoam_bashrc=${DSMCFOAM_BASHRC_USED:-module-path}"
    echo "dsmcFoam=$(command -v dsmcFoam)"
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
    [[ -f "log.${CURRENT_STAGE}" ]] && tail -120 "log.${CURRENT_STAGE}" >&2
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
    grep -Fq 'nEquivalentParticles    0.197507271273125;' constant/dsmcProperties
    grep -Fq 'hex (0 1 2 3 4 5 6 7) (80 80 1)' system/blockMeshDict
    grep -Fq 'deltaT              5e-12;' system/controlDict
    grep -Fq 'endTime             1e-6;' system/controlDict
    grep -Fq 'timeStart       2e-7;' system/controlDict
    grep -Fq 'writeInterval       20000;' system/controlDict
    grep -Fq "numberOfSubdomains ${MPI_NTASKS};" system/decomposeParDict
    grep -Fq "n       (${PROC_X} ${PROC_Y} 1);" system/decomposeParDict
    echo 'KN01_HQ_CASE_CONTRACT_PASS'
}

verify_parallel_initial_state() {
    local rank proc field
    local -a required_fields=(boundaryT boundaryU dsmcRhoN fD iDof internalE linearKE momentum q rhoM rhoN)
    for ((rank=0; rank<MPI_NTASKS; rank++)); do
        proc="processor${rank}"
        [[ -d "$proc/0" ]] || { echo "ERROR: missing $proc/0" >&2; return 1; }
        for field in "${required_fields[@]}"; do
            [[ -f "$proc/0/$field" ]] || { echo "ERROR: missing $proc/0/$field" >&2; return 1; }
        done
    done
}

run_stage caseContract verify_case_contract
run_stage blockMesh blockMesh
run_stage checkMesh checkMesh
run_stage dsmcInitialise dsmcInitialise
run_stage decomposePar decomposePar -force
run_stage initialState verify_parallel_initial_state
run_stage dsmcFoam mpirun -np "$MPI_NTASKS" dsmcFoam -parallel
run_stage reconstructPar reconstructPar -latestTime

CURRENT_STAGE="finalize"
{
    echo "finished_utc=$(date -u +%FT%TZ)"
    echo "status=COMPLETED"
} | tee -a RUN_METADATA.txt
touch "$CAMPAIGN_ROOT/COMPLETED.member_${SLURM_ARRAY_TASK_ID}"
if [[ $(find "$CAMPAIGN_ROOT" -maxdepth 1 -type f -name 'COMPLETED.member_*' | wc -l) -eq 3 ]]; then
    touch "$CAMPAIGN_ROOT/CAMPAIGN_COMPLETED"
fi
trap - EXIT

