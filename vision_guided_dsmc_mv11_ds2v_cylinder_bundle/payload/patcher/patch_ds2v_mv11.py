#!/usr/bin/env python3
"""Fail-closed DS2V patcher for the MV11 hypersonic-cylinder campaign.

The patch is intentionally source-based: Bird's complete DS2V source is not
redistributed.  It adds restart-safe intrinsic RNG control, guards LOG(RANF)
against an exact zero, and records additive raw kinetic moments needed to
reconstruct stress and heat-flux fields without subtracting pre-averaged
macroscopic outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


RNG_MODULE = r'''MODULE RNG_CONTROL
IMPLICIT NONE
INTEGER(KIND=8) :: RNG_BASE_SEED=-1_8
CONTAINS

SUBROUTINE INIT_RANDOM_SEED(IRUN)
IMPLICIT NONE
INTEGER, INTENT(IN) :: IRUN
INTEGER :: N,I,IOS,NSAVED,VERSION
INTEGER, ALLOCATABLE :: SEEDV(:)
INTEGER(KIND=8) :: X
INTEGER(KIND=8), PARAMETER :: MODULUS=2147483647_8
LOGICAL :: EXISTS
CHARACTER(LEN=32) :: MAGIC

CALL RANDOM_SEED(SIZE=N)
ALLOCATE(SEEDV(N))

IF (IRUN == 3) THEN
  INQUIRE(FILE='RANDOM_SEED.IN',EXIST=EXISTS)
  IF (.NOT. EXISTS) STOP 'FATAL: RANDOM_SEED.IN required for IRUN=3.'
  OPEN (98,FILE='RANDOM_SEED.IN',STATUS='OLD',FORM='FORMATTED',IOSTAT=IOS)
  IF (IOS /= 0) STOP 'FATAL: cannot open RANDOM_SEED.IN'
  READ (98,*,IOSTAT=IOS) RNG_BASE_SEED
  CLOSE (98)
  IF ((IOS /= 0).OR.(RNG_BASE_SEED <= 0_8)) &
    STOP 'FATAL: RANDOM_SEED.IN must contain one positive integer.'

  X=MOD(RNG_BASE_SEED,MODULUS)
  IF (X == 0_8) X=1_8
  DO I=1,N
    X=MOD(1103515245_8*X+12345_8,MODULUS)
    SEEDV(I)=INT(MOD(X+104729_8*I,MODULUS-1_8)+1_8)
  END DO
  CALL RANDOM_SEED(PUT=SEEDV)

  OPEN (98,FILE='RNG_SEED_USED.txt',STATUS='REPLACE',FORM='FORMATTED')
  WRITE (98,'(A)') 'RNG_CONTROL_VERSION=1'
  WRITE (98,'(A)') 'MODE=NEW_RUN'
  WRITE (98,'(A,I0)') 'BASE_SEED=',RNG_BASE_SEED
  WRITE (98,'(A,I0)') 'RANDOM_SEED_SIZE=',N
  WRITE (98,'(A)') 'SEED_VECTOR_BEGIN'
  WRITE (98,*) SEEDV
  WRITE (98,'(A)') 'SEED_VECTOR_END'
  CLOSE (98)
ELSE
  INQUIRE(FILE='RNG_STATE.DAT',EXIST=EXISTS)
  IF (.NOT. EXISTS) STOP 'FATAL: RNG_STATE.DAT missing for IRUN=1/2.'
  OPEN (98,FILE='RNG_STATE.DAT',STATUS='OLD',FORM='FORMATTED',IOSTAT=IOS)
  IF (IOS /= 0) STOP 'FATAL: cannot open RNG_STATE.DAT'
  READ (98,'(A)',IOSTAT=IOS) MAGIC
  IF ((IOS /= 0).OR.(TRIM(MAGIC) /= 'DS2V_RNG_STATE')) &
    STOP 'FATAL: invalid RNG_STATE.DAT header.'
  READ (98,*,IOSTAT=IOS) VERSION
  IF ((IOS /= 0).OR.(VERSION /= 1)) STOP 'FATAL: unsupported RNG state.'
  READ (98,*,IOSTAT=IOS) NSAVED
  IF ((IOS /= 0).OR.(NSAVED /= N)) &
    STOP 'FATAL: RNG state/compiler seed-size mismatch.'
  READ (98,*,IOSTAT=IOS) RNG_BASE_SEED
  READ (98,*,IOSTAT=IOS) SEEDV
  CLOSE (98)
  IF (IOS /= 0) STOP 'FATAL: cannot read RNG state vector.'
  CALL RANDOM_SEED(PUT=SEEDV)

  OPEN (98,FILE='RNG_SEED_USED.txt',STATUS='UNKNOWN',POSITION='APPEND',FORM='FORMATTED')
  WRITE (98,'(A,I0,A,I0)') 'MODE=RESUME, IRUN=',IRUN,', RANDOM_SEED_SIZE=',N
  CLOSE (98)
END IF

DEALLOCATE(SEEDV)
END SUBROUTINE INIT_RANDOM_SEED

SUBROUTINE SAVE_RANDOM_STATE
IMPLICIT NONE
INTEGER :: N,IOS
INTEGER, ALLOCATABLE :: SEEDV(:)
CALL RANDOM_SEED(SIZE=N)
ALLOCATE(SEEDV(N))
CALL RANDOM_SEED(GET=SEEDV)
OPEN (98,FILE='RNG_STATE.DAT',STATUS='REPLACE',FORM='FORMATTED',IOSTAT=IOS)
IF (IOS /= 0) STOP 'FATAL: cannot write RNG_STATE.DAT'
WRITE (98,'(A)') 'DS2V_RNG_STATE'
WRITE (98,*) 1
WRITE (98,*) N
WRITE (98,*) RNG_BASE_SEED
WRITE (98,*) SEEDV
CLOSE (98)
DEALLOCATE(SEEDV)
END SUBROUTINE SAVE_RANDOM_STATE

END MODULE RNG_CONTROL

!*****************************************************************************

'''


MOMENT_MODULE = r'''MODULE MV11_KINETIC_MOMENTS
IMPLICIT NONE
INTEGER, PARAMETER :: MV11_NMOM=13
REAL(KIND=8), ALLOCATABLE :: MV11_SUM(:,:,:)
INTEGER(KIND=8) :: MV11_BLOCK_SAMPLES=0_8
CONTAINS

SUBROUTINE MV11_INITIALIZE(NCELLS,NSPECIES)
IMPLICIT NONE
INTEGER, INTENT(IN) :: NCELLS,NSPECIES
IF (.NOT. ALLOCATED(MV11_SUM)) THEN
  ALLOCATE(MV11_SUM(MV11_NMOM,NCELLS,NSPECIES))
  MV11_SUM=0.D0
  MV11_BLOCK_SAMPLES=0_8
ELSE IF ((SIZE(MV11_SUM,2) /= NCELLS).OR. &
         (SIZE(MV11_SUM,3) /= NSPECIES)) THEN
  STOP 'FATAL: MV11 moment dimensions changed after allocation.'
END IF
END SUBROUTINE MV11_INITIALIZE

SUBROUTINE MV11_BEGIN_SAMPLE(NCELLS,NSPECIES)
IMPLICIT NONE
INTEGER, INTENT(IN) :: NCELLS,NSPECIES
CALL MV11_INITIALIZE(NCELLS,NSPECIES)
MV11_BLOCK_SAMPLES=MV11_BLOCK_SAMPLES+1_8
END SUBROUTINE MV11_BEGIN_SAMPLE

SUBROUTINE MV11_ACCUMULATE(NC,LS,WF,U,V,W,MASS)
IMPLICIT NONE
INTEGER, INTENT(IN) :: NC,LS
REAL, INTENT(IN) :: WF,U,V,W,MASS
REAL(KIND=8) :: DW,DU,DV,DWVEL,DMASS,ENERGY
IF (.NOT. ALLOCATED(MV11_SUM)) STOP 'FATAL: MV11 moments not initialized.'
IF ((NC < 1).OR.(NC > SIZE(MV11_SUM,2))) STOP 'FATAL: MV11 illegal cell.'
IF ((LS < 1).OR.(LS > SIZE(MV11_SUM,3))) STOP 'FATAL: MV11 illegal species.'
DW=REAL(WF,KIND=8)
DU=REAL(U,KIND=8)
DV=REAL(V,KIND=8)
DWVEL=REAL(W,KIND=8)
DMASS=REAL(MASS,KIND=8)
ENERGY=0.5D0*DMASS*(DU*DU+DV*DV+DWVEL*DWVEL)
MV11_SUM(1,NC,LS)=MV11_SUM(1,NC,LS)+DW
MV11_SUM(2,NC,LS)=MV11_SUM(2,NC,LS)+DW*DU
MV11_SUM(3,NC,LS)=MV11_SUM(3,NC,LS)+DW*DV
MV11_SUM(4,NC,LS)=MV11_SUM(4,NC,LS)+DW*DWVEL
MV11_SUM(5,NC,LS)=MV11_SUM(5,NC,LS)+DW*DU*DU
MV11_SUM(6,NC,LS)=MV11_SUM(6,NC,LS)+DW*DV*DV
MV11_SUM(7,NC,LS)=MV11_SUM(7,NC,LS)+DW*DWVEL*DWVEL
MV11_SUM(8,NC,LS)=MV11_SUM(8,NC,LS)+DW*DU*DV
MV11_SUM(9,NC,LS)=MV11_SUM(9,NC,LS)+DW*DU*DWVEL
MV11_SUM(10,NC,LS)=MV11_SUM(10,NC,LS)+DW*DV*DWVEL
MV11_SUM(11,NC,LS)=MV11_SUM(11,NC,LS)+DW*ENERGY
MV11_SUM(12,NC,LS)=MV11_SUM(12,NC,LS)+DW*ENERGY*DU
MV11_SUM(13,NC,LS)=MV11_SUM(13,NC,LS)+DW*ENERGY*DV
END SUBROUTINE MV11_ACCUMULATE

SUBROUTINE MV11_RESET
IMPLICIT NONE
IF (ALLOCATED(MV11_SUM)) MV11_SUM=0.D0
MV11_BLOCK_SAMPLES=0_8
END SUBROUTINE MV11_RESET

SUBROUTINE MV11_WRITE_BLOCK(NOUT,TIME,SFAC,FNUM,NSAMP,NCELLS,NSPECIES,CELL_ARRAY)
IMPLICIT NONE
INTEGER, INTENT(IN) :: NOUT,NSAMP,NCELLS,NSPECIES
! Bird's locked DS2V source declares TIME as REAL(KIND=8), while SFAC and
! FNUM are default REAL.  Keep the explicit module interface kind-exact.
REAL(KIND=8), INTENT(IN) :: TIME
REAL, INTENT(IN) :: SFAC,FNUM
REAL, INTENT(IN) :: CELL_ARRAY(:,:)
INTEGER :: UNIT,N,L
REAL(KIND=8) :: X,Y,AREA
CHARACTER(LEN=64) :: FILENAME
IF (.NOT. ALLOCATED(MV11_SUM)) RETURN
IF (MV11_BLOCK_SAMPLES <= 0_8) RETURN
IF ((SIZE(CELL_ARRAY,1) < 3).OR.(SIZE(CELL_ARRAY,2) < NCELLS)) &
  STOP 'FATAL: MV11 CELL array shape mismatch.'
WRITE (FILENAME,'("MV11_MOMENTS_NOUT",I4.4,".DAT")') NOUT
OPEN (NEWUNIT=UNIT,FILE=TRIM(FILENAME),STATUS='REPLACE',FORM='FORMATTED')
WRITE (UNIT,'(A)') '# MV11_ADDITIVE_KINETIC_MOMENTS_VERSION=1'
WRITE (UNIT,'(A,I0,A,ES24.16,A,ES24.16,A,ES24.16,A,I0,A,I0)') &
  '# NOUT=',NOUT,' TIME=',REAL(TIME/SFAC,KIND=8), &
  ' SFAC=',REAL(SFAC,KIND=8),' FNUM=',REAL(FNUM,KIND=8), &
  ' NSAMP_DS2V=',NSAMP,' BLOCK_SAMPLES=',MV11_BLOCK_SAMPLES
WRITE (UNIT,'(A)') &
  '# cell species x_m y_m area_m2 m0 m1x m1y m1z m2xx m2yy m2zz m2xy m2xz m2yz energy energy_vx energy_vy'
DO L=1,NSPECIES
  DO N=1,NCELLS
    IF (MV11_SUM(1,N,L) > 0.D0) THEN
      X=REAL(CELL_ARRAY(2,N)/SFAC,KIND=8)
      Y=REAL(CELL_ARRAY(3,N)/SFAC,KIND=8)
      AREA=REAL(CELL_ARRAY(1,N)/(SFAC*SFAC),KIND=8)
      WRITE (UNIT,*) N,L,X,Y,AREA,MV11_SUM(:,N,L)
    END IF
  END DO
END DO
CLOSE (UNIT)
CALL MV11_RESET
END SUBROUTINE MV11_WRITE_BLOCK

END MODULE MV11_KINETIC_MOMENTS

!*****************************************************************************

'''


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def subroutine_span(text: str, name: str) -> tuple[int, int]:
    start_match = re.search(rf"(?mi)^\s*SUBROUTINE\s+{re.escape(name)}\b", text)
    if not start_match:
        raise RuntimeError(f"required subroutine {name} was not found")
    end_match = re.search(
        rf"(?mi)^\s*END\s+SUBROUTINE\s+{re.escape(name)}\s*$",
        text[start_match.start() :],
    )
    if not end_match:
        raise RuntimeError(f"end of subroutine {name} was not found")
    return start_match.start(), start_match.start() + end_match.end()


def add_use(block: str, module: str) -> str:
    if re.search(rf"(?mi)^\s*USE\s+{re.escape(module)}\s*$", block):
        raise RuntimeError(f"{module} already appears in target subroutine")
    implicit = re.search(r"(?mi)^\s*IMPLICIT\s+NONE\s*$", block)
    if implicit:
        return block[: implicit.start()] + f"USE {module}\n" + block[implicit.start() :]
    use_matches = list(re.finditer(r"(?mi)^\s*USE\s+\w+\s*$", block))
    if not use_matches:
        raise RuntimeError("neither IMPLICIT NONE nor a USE anchor was found")
    insertion = use_matches[-1].end()
    return block[:insertion] + f"\nUSE {module}" + block[insertion:]


def replace_block(text: str, name: str, transform) -> str:
    start, end = subroutine_span(text, name)
    return text[:start] + transform(text[start:end]) + text[end:]


def patch_source(source: Path, output: Path, report_path: Path) -> None:
    source_bytes = source.read_bytes()
    text = source_bytes.decode("utf-8-sig", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if "MODULE RNG_CONTROL" in text or "MODULE MV11_KINETIC_MOMENTS" in text:
        raise RuntimeError("source already contains an MV11 control module")

    program_matches = list(re.finditer(r"(?mi)^\s*PROGRAM\s+DS2V\s*$", text))
    if len(program_matches) != 1:
        raise RuntimeError(f"expected one PROGRAM DS2V, found {len(program_matches)}")
    insertion = program_matches[0].start()
    text = text[:insertion] + RNG_MODULE + MOMENT_MODULE + text[insertion:]

    main_start = text.index("PROGRAM DS2V", insertion)
    main_implicit = text.index("IMPLICIT NONE", main_start)
    main_use = text[main_start:main_implicit]
    if main_use.count("USE MOLFILE") != 1:
        raise RuntimeError("unique USE MOLFILE main-program anchor was not found")
    main_use = main_use.replace(
        "USE MOLFILE", "USE MOLFILE\nUSE RNG_CONTROL", 1
    )
    text = text[:main_start] + main_use + text[main_implicit:]

    irun_anchor = "IF (IRUN == 3) WRITE (9,*) 'Starting a new run'"
    if text.count(irun_anchor) != 1:
        raise RuntimeError("unique IRUN-selection anchor was not found")
    text = text.replace(
        irun_anchor, irun_anchor + "\nCALL INIT_RANDOM_SEED(IRUN)", 1
    )

    restart_insertions = 0
    restart_calls_found = 0
    for argument in ("0", "1"):
        pattern = re.compile(
            rf"(?mi)^(\s*)CALL\s+WRITE_RESTART\s*\(\s*{argument}\s*\)\s*$"
        )
        matches = list(pattern.finditer(text))
        restart_calls_found += len(matches)
        restart_insertions += len(matches)
        text = pattern.sub(
            lambda match: match.group(0)
            + "\n"
            + match.group(1)
            + "CALL SAVE_RANDOM_STATE",
            text,
        )
    if restart_calls_found == 0 and "RESTART_WRITE_DISABLED_FRESH_ONLY" not in text:
        raise RuntimeError(
            "no WRITE_RESTART calls and no fresh-only restart-disable marker were found"
        )

    unsafe_log = re.compile(r"(?i)LOG\s*\(\s*RANF\s*\)")
    text, rng_guards = unsafe_log.subn(
        "LOG(MAX(RANF,0.5*EPSILON(RANF)))", text
    )
    if unsafe_log.search(text):
        raise RuntimeError("an unsafe LOG(RANF) call remains")

    def patch_sample(block: str) -> str:
        block = add_use(block, "MV11_KINETIC_MOMENTS")
        sample_anchor = "NSAMP=NSAMP+1"
        if block.count(sample_anchor) != 1:
            raise RuntimeError("unique NSAMP increment was not found in SAMPLE_FLOW")
        block = block.replace(
            sample_anchor,
            sample_anchor + "\nCALL MV11_BEGIN_SAMPLE(NCELLS,MSP)",
            1,
        )
        weight_anchor = "IF (IWF == 1) WF=1.+PP(2,N)*RWF"
        if block.count(weight_anchor) != 1:
            raise RuntimeError("unique sampling-weight anchor was not found")
        return block.replace(
            weight_anchor,
            weight_anchor
            + "\n    CALL MV11_ACCUMULATE(NC,LS,WF,PV(1,N),PV(2,N),PV(3,N),SP(5,LS))",
            1,
        )

    text = replace_block(text, "SAMPLE_FLOW", patch_sample)

    def patch_output(block: str) -> str:
        block = add_use(block, "MV11_KINETIC_MOMENTS")
        anchor = "WRITE (*,*) 'Output files written'"
        if block.count(anchor) != 1:
            raise RuntimeError("unique output-complete anchor was not found")
        return block.replace(
            anchor,
            "CALL MV11_WRITE_BLOCK(NOUT,TIME,SFAC,FNUM,NSAMP,NCELLS,MSP,CELL)\n"
            + anchor,
            1,
        )

    text = replace_block(text, "OUTPUT_RESULTS", patch_output)

    resets = 0
    for name in ("INITIALISE_SAMPLES", "INITIALISE_SAMPLES101"):
        def patch_reset(block: str) -> str:
            nonlocal resets
            block = add_use(block, "MV11_KINETIC_MOMENTS")
            anchor = "CS=0. ; CSS=0. ; CSSS=0. ; CSSO=0. ; CSSOG=0."
            if block.count(anchor) != 1:
                raise RuntimeError(f"unique CS reset was not found in {name}")
            resets += 1
            return block.replace(anchor, anchor + "\nCALL MV11_RESET", 1)

        text = replace_block(text, name, patch_reset)

    required = (
        "CALL INIT_RANDOM_SEED(IRUN)",
        "CALL MV11_BEGIN_SAMPLE(NCELLS,MSP)",
        "CALL MV11_ACCUMULATE",
        "CALL MV11_WRITE_BLOCK",
        "MV11_ADDITIVE_KINETIC_MOMENTS_VERSION=1",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(f"post-patch markers missing: {missing}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    output_bytes = output.read_bytes()
    report = {
        "patch_version": 2,
        "source": str(source),
        "output": str(output),
        "source_sha256": sha256_bytes(source_bytes),
        "output_sha256": sha256_bytes(output_bytes),
        "restart_state_save_insertions": restart_insertions,
        "restart_mode": (
            "restart_safe" if restart_calls_found else "fresh_only_source"
        ),
        "rng_zero_guards_inserted": rng_guards,
        "sample_reset_insertions": resets,
        "moment_names": [
            "m0",
            "m1x",
            "m1y",
            "m1z",
            "m2xx",
            "m2yy",
            "m2zz",
            "m2xy",
            "m2xz",
            "m2yz",
            "energy",
            "energy_vx",
            "energy_vy",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    patch_source(args.source.resolve(), args.out.resolve(), args.report.resolve())


if __name__ == "__main__":
    main()
