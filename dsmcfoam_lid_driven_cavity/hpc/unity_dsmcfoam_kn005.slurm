#!/usr/bin/env bash
#SBATCH --job-name=dsmcfoam-kn005
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --constraint=avx512
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/dsmcfoam-kn005-%j.out
#SBATCH --error=logs/dsmcfoam-kn005-%j.err

set -euo pipefail

CASE_ROOT="${CASE_ROOT:?CASE_ROOT was not exported by run_unity.sh}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT was not exported by run_unity.sh}"
LAST_ENV="${LAST_ENV:?LAST_ENV was not exported by run_unity.sh}"

source_openfoam_environment() {
    local root rc
    local -a roots=()

    [[ -n "${EBROOTOPENFOAM:-}" ]] && roots+=("$EBROOTOPENFOAM")
    [[ -n "${WM_PROJECT_DIR:-}" ]] && roots+=("$WM_PROJECT_DIR")

    for root in "${roots[@]}"; do
        for rc in \
            "$root/etc/bashrc" \
            "$root/OpenFOAM-v2406/etc/bashrc" \
            "$root/OpenFOAM-v2006/etc/bashrc"; do
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

    if [[ -n "${DSMCFOAM_BASHRC:-}" && -r "${DSMCFOAM_BASHRC}" ]]; then
        set +u
        source "${DSMCFOAM_BASHRC}"
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

        if command -v dsmcFoam >/dev/null 2>&1; then
            return 0
        fi

        # Unity's EasyBuild module can expose the installation root without
        # sourcing OpenFOAM's etc/bashrc. Source it inside the AVX-512 job.
        if source_openfoam_environment; then
            return 0
        fi
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
    for root in "${EBROOTOPENFOAM:-}" "${WM_PROJECT_DIR:-}"; do
        [[ -d "$root" ]] || continue
        echo "--- candidates under $root ---" >&2
        find "$root" -maxdepth 8 \
            \( -type f -o -type l \) \
            \( -name dsmcFoam -o -name dsmcInitialise -o -path '*/etc/bashrc' \) \
            -print 2>/dev/null | head -80 >&2 || true
    done
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
RUN_DIR="$RUN_ROOT/${SLURM_JOB_ID}_kn005"
if [[ -e "$RUN_DIR" ]]; then
    echo "ERROR: run directory already exists: $RUN_DIR" >&2
    exit 2
fi
cp -a "$CASE_ROOT" "$RUN_DIR"
cd "$RUN_DIR"

{
    echo "job_id=$SLURM_JOB_ID"
    echo "host=$(hostname)"
    echo "started_utc=$(date -u +%FT%TZ)"
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

run_stage blockMesh blockMesh
run_stage checkMesh checkMesh
run_stage dsmcInitialise dsmcInitialise
run_stage decomposePar decomposePar -force
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
    printf 'export SLURM_LOG=%q\n' "${SLURM_LOG_PATH:-${CASE_ROOT%/case}/logs/dsmcfoam-kn005-${SLURM_JOB_ID}.out}"
} > "$tmp_env"
mv "$tmp_env" "$LAST_ENV"
trap - EXIT
