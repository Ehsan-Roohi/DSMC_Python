#!/usr/bin/env bash
#SBATCH --job-name=dsmcfoam-kn005
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/dsmcfoam-kn005-%j.out
#SBATCH --error=logs/dsmcfoam-kn005-%j.err

set -euo pipefail

CASE_ROOT="${CASE_ROOT:?CASE_ROOT was not exported by run_unity.sh}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT was not exported by run_unity.sh}"
LAST_ENV="${LAST_ENV:?LAST_ENV was not exported by run_unity.sh}"

load_openfoam() {
    if command -v dsmcFoam >/dev/null 2>&1; then
        return 0
    fi

    set +u
    if [[ -n "${DSMCFOAM_BASHRC:-}" && -r "${DSMCFOAM_BASHRC}" ]]; then
        source "${DSMCFOAM_BASHRC}"
    fi
    set -u
    if command -v dsmcFoam >/dev/null 2>&1; then
        return 0
    fi

    if command -v module >/dev/null 2>&1; then
        local candidate parent
        # Unity uses hierarchical Lmod modules: the foss parent toolchain must
        # be loaded before the corresponding OpenFOAM module becomes visible.
        while read -r parent candidate; do
            module purge >/dev/null 2>&1 || true
            if [[ "$parent" != "none" ]]; then
                module load "$parent" >/dev/null 2>&1 || continue
            fi
            if module load "$candidate" >/dev/null 2>&1; then
                if command -v dsmcFoam >/dev/null 2>&1; then
                    export DSMCFOAM_MODULE="$candidate"
                    return 0
                fi
                module unload "$candidate" >/dev/null 2>&1 || true
            fi
        done <<'MODULE_CANDIDATES'
foss/2023a OpenFOAM/v2406-foss-2023a
foss/2020a OpenFOAM/v2006-foss-2020a
none OpenFOAM/10
none openfoam/10
none OpenFOAM/v2306
none openfoam/2306
none OpenFOAM
none openfoam
MODULE_CANDIDATES
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
            return 0
        fi
    done
    shopt -u nullglob
    return 1
}

if ! load_openfoam; then
    echo "ERROR: dsmcFoam was not found." >&2
    echo "Resubmit with DSMCFOAM_BASHRC=/absolute/path/to/OpenFOAM/etc/bashrc." >&2
    if command -v module >/dev/null 2>&1; then
        module -t spider OpenFOAM 2>&1 || true
        module -t avail 2>&1 | grep -i openfoam || true
    fi
    exit 127
fi

for command_name in blockMesh checkMesh dsmcInitialise decomposePar dsmcFoam reconstructPar mpirun; do
    command -v "$command_name" >/dev/null 2>&1 || {
        echo "ERROR: required command not found: $command_name" >&2
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
    echo "dsmcFoam=$(command -v dsmcFoam)"
    command -v foamVersion >/dev/null 2>&1 && foamVersion || true
} | tee RUN_METADATA.txt

blockMesh > log.blockMesh 2>&1
checkMesh > log.checkMesh 2>&1
dsmcInitialise > log.dsmcInitialise 2>&1
decomposePar -force > log.decomposePar 2>&1

mpirun -np "$SLURM_NTASKS" dsmcFoam -parallel > log.dsmcFoam 2>&1
reconstructPar -latestTime > log.reconstructPar 2>&1

{
    echo "finished_utc=$(date -u +%FT%TZ)"
    echo "status=COMPLETED"
} | tee -a RUN_METADATA.txt

tmp_env="${LAST_ENV}.tmp.${SLURM_JOB_ID}"
{
    printf 'export JOB_ID=%q\n' "$SLURM_JOB_ID"
    printf 'export CASE_DIR=%q\n' "$RUN_DIR"
    printf 'export SLURM_LOG=%q\n' "${CASE_ROOT%/case}/logs/dsmcfoam-kn005-${SLURM_JOB_ID}.out"
} > "$tmp_env"
mv "$tmp_env" "$LAST_ENV"
