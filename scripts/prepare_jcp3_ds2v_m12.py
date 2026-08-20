#!/usr/bin/env python3
"""Prepare a fail-fast Mach-12 DS2V cylinder pilot from the locked Mach-10 case.

The Bird/DS2V source and ``DS2VD.DAT`` remain external inputs.  This script
adds deterministic fresh-run seeding, additive kinetic-moment output, a
per-output wall-tally file, and an optional one-output pilot stop.  It also
changes only the locked freestream-x-velocity token from Mach 10 to Mach 12.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


M10_SPEED = 2634.1
M12_SPEED = 3160.92
FLOAT_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?(?![A-Za-z0-9_.])"
)


MODULES = r"""
MODULE JCP3_RNG_CONTROL
IMPLICIT NONE
INTEGER(KIND=8) :: JCP3_BASE_SEED=-1_8
CONTAINS

SUBROUTINE JCP3_INIT_RANDOM_SEED(IRUN)
IMPLICIT NONE
INTEGER, INTENT(IN) :: IRUN
INTEGER :: N,I,IOS
INTEGER, ALLOCATABLE :: SEEDV(:)
INTEGER(KIND=8) :: X
INTEGER(KIND=8), PARAMETER :: MODULUS=2147483647_8
LOGICAL :: EXISTS
CALL RANDOM_SEED(SIZE=N)
ALLOCATE(SEEDV(N))
IF (IRUN /= 3) STOP 'FATAL: JCP3 pilot accepts fresh IRUN=3 only.'
INQUIRE(FILE='RANDOM_SEED.IN',EXIST=EXISTS)
IF (.NOT. EXISTS) STOP 'FATAL: RANDOM_SEED.IN required for JCP3.'
OPEN (98,FILE='RANDOM_SEED.IN',STATUS='OLD',FORM='FORMATTED',IOSTAT=IOS)
IF (IOS /= 0) STOP 'FATAL: cannot open RANDOM_SEED.IN'
READ (98,*,IOSTAT=IOS) JCP3_BASE_SEED
CLOSE (98)
IF ((IOS /= 0).OR.(JCP3_BASE_SEED <= 0_8)) &
  STOP 'FATAL: RANDOM_SEED.IN must contain one positive integer.'
X=MOD(JCP3_BASE_SEED,MODULUS)
IF (X == 0_8) X=1_8
DO I=1,N
  X=MOD(1103515245_8*X+12345_8,MODULUS)
  SEEDV(I)=INT(MOD(X+104729_8*I,MODULUS-1_8)+1_8)
END DO
CALL RANDOM_SEED(PUT=SEEDV)
OPEN (98,FILE='RNG_SEED_USED.txt',STATUS='REPLACE',FORM='FORMATTED')
WRITE (98,'(A)') 'JCP3_RNG_CONTROL_VERSION=1'
WRITE (98,'(A,I0)') 'BASE_SEED=',JCP3_BASE_SEED
WRITE (98,'(A,I0)') 'RANDOM_SEED_SIZE=',N
WRITE (98,*) SEEDV
CLOSE (98)
DEALLOCATE(SEEDV)
END SUBROUTINE JCP3_INIT_RANDOM_SEED

END MODULE JCP3_RNG_CONTROL

!*****************************************************************************

MODULE JCP3_KINETIC_MOMENTS
IMPLICIT NONE
INTEGER, PARAMETER :: JCP3_NMOM=13
REAL(KIND=8), ALLOCATABLE :: JCP3_SUM(:,:,:)
INTEGER(KIND=8) :: JCP3_BLOCK_SAMPLES=0_8
CONTAINS

SUBROUTINE JCP3_INITIALIZE(NCELLS,NSPECIES)
IMPLICIT NONE
INTEGER, INTENT(IN) :: NCELLS,NSPECIES
IF (.NOT. ALLOCATED(JCP3_SUM)) THEN
  ALLOCATE(JCP3_SUM(JCP3_NMOM,NCELLS,NSPECIES))
  JCP3_SUM=0.D0
  JCP3_BLOCK_SAMPLES=0_8
ELSE IF ((SIZE(JCP3_SUM,2) /= NCELLS).OR. &
         (SIZE(JCP3_SUM,3) /= NSPECIES)) THEN
  STOP 'FATAL: JCP3 moment dimensions changed after allocation.'
END IF
END SUBROUTINE JCP3_INITIALIZE

SUBROUTINE JCP3_BEGIN_SAMPLE(NCELLS,NSPECIES)
IMPLICIT NONE
INTEGER, INTENT(IN) :: NCELLS,NSPECIES
CALL JCP3_INITIALIZE(NCELLS,NSPECIES)
JCP3_BLOCK_SAMPLES=JCP3_BLOCK_SAMPLES+1_8
END SUBROUTINE JCP3_BEGIN_SAMPLE

SUBROUTINE JCP3_ACCUMULATE(NC,LS,WF,U,V,W,MASS)
IMPLICIT NONE
INTEGER, INTENT(IN) :: NC,LS
REAL, INTENT(IN) :: WF,U,V,W,MASS
REAL(KIND=8) :: DW,DU,DV,DWVEL,DMASS,ENERGY
IF (.NOT. ALLOCATED(JCP3_SUM)) STOP 'FATAL: JCP3 moments not initialized.'
IF ((NC < 1).OR.(NC > SIZE(JCP3_SUM,2))) STOP 'FATAL: JCP3 illegal cell.'
IF ((LS < 1).OR.(LS > SIZE(JCP3_SUM,3))) STOP 'FATAL: JCP3 illegal species.'
DW=REAL(WF,KIND=8)
DU=REAL(U,KIND=8)
DV=REAL(V,KIND=8)
DWVEL=REAL(W,KIND=8)
DMASS=REAL(MASS,KIND=8)
ENERGY=0.5D0*DMASS*(DU*DU+DV*DV+DWVEL*DWVEL)
JCP3_SUM(1,NC,LS)=JCP3_SUM(1,NC,LS)+DW
JCP3_SUM(2,NC,LS)=JCP3_SUM(2,NC,LS)+DW*DU
JCP3_SUM(3,NC,LS)=JCP3_SUM(3,NC,LS)+DW*DV
JCP3_SUM(4,NC,LS)=JCP3_SUM(4,NC,LS)+DW*DWVEL
JCP3_SUM(5,NC,LS)=JCP3_SUM(5,NC,LS)+DW*DU*DU
JCP3_SUM(6,NC,LS)=JCP3_SUM(6,NC,LS)+DW*DV*DV
JCP3_SUM(7,NC,LS)=JCP3_SUM(7,NC,LS)+DW*DWVEL*DWVEL
JCP3_SUM(8,NC,LS)=JCP3_SUM(8,NC,LS)+DW*DU*DV
JCP3_SUM(9,NC,LS)=JCP3_SUM(9,NC,LS)+DW*DU*DWVEL
JCP3_SUM(10,NC,LS)=JCP3_SUM(10,NC,LS)+DW*DV*DWVEL
JCP3_SUM(11,NC,LS)=JCP3_SUM(11,NC,LS)+DW*ENERGY
JCP3_SUM(12,NC,LS)=JCP3_SUM(12,NC,LS)+DW*ENERGY*DU
JCP3_SUM(13,NC,LS)=JCP3_SUM(13,NC,LS)+DW*ENERGY*DV
END SUBROUTINE JCP3_ACCUMULATE

SUBROUTINE JCP3_RESET
IMPLICIT NONE
IF (ALLOCATED(JCP3_SUM)) JCP3_SUM=0.D0
JCP3_BLOCK_SAMPLES=0_8
END SUBROUTINE JCP3_RESET

SUBROUTINE JCP3_WRITE_BLOCK(NOUT,TIME,SFAC,FNUM,NSAMP,NCELLS,NSPECIES,CELL_ARRAY)
IMPLICIT NONE
INTEGER, INTENT(IN) :: NOUT,NSAMP,NCELLS,NSPECIES
REAL(KIND=8), INTENT(IN) :: TIME
REAL, INTENT(IN) :: SFAC,FNUM
REAL, INTENT(IN) :: CELL_ARRAY(:,:)
INTEGER :: UNIT,N,L
REAL(KIND=8) :: X,Y,AREA
CHARACTER(LEN=64) :: FILENAME
IF (.NOT. ALLOCATED(JCP3_SUM)) RETURN
IF (JCP3_BLOCK_SAMPLES <= 0_8) RETURN
IF ((SIZE(CELL_ARRAY,1) < 3).OR.(SIZE(CELL_ARRAY,2) < NCELLS)) &
  STOP 'FATAL: JCP3 CELL array shape mismatch.'
WRITE (FILENAME,'("JCP3_MOMENTS_NOUT",I4.4,".DAT")') NOUT
OPEN (NEWUNIT=UNIT,FILE=TRIM(FILENAME),STATUS='REPLACE',FORM='FORMATTED')
WRITE (UNIT,'(A)') '# JCP3_ADDITIVE_KINETIC_MOMENTS_VERSION=1'
WRITE (UNIT,'(A,I0,A,ES24.16,A,ES24.16,A,ES24.16,A,I0,A,I0)') &
  '# NOUT=',NOUT,' TIME=',REAL(TIME/SFAC,KIND=8), &
  ' SFAC=',REAL(SFAC,KIND=8),' FNUM=',REAL(FNUM,KIND=8), &
  ' NSAMP_DS2V=',NSAMP,' BLOCK_SAMPLES=',JCP3_BLOCK_SAMPLES
WRITE (UNIT,'(A)') &
  '# cell species x_m y_m area_m2 m0 m1x m1y m1z m2xx m2yy m2zz m2xy m2xz m2yz energy energy_vx energy_vy'
DO L=1,NSPECIES
  DO N=1,NCELLS
    IF (JCP3_SUM(1,N,L) > 0.D0) THEN
      X=REAL(CELL_ARRAY(2,N)/SFAC,KIND=8)
      Y=REAL(CELL_ARRAY(3,N)/SFAC,KIND=8)
      AREA=REAL(CELL_ARRAY(1,N)/(SFAC*SFAC),KIND=8)
      WRITE (UNIT,*) N,L,X,Y,AREA,JCP3_SUM(:,N,L)
    END IF
  END DO
END DO
CLOSE (UNIT)
CALL JCP3_RESET
END SUBROUTINE JCP3_WRITE_BLOCK

SUBROUTINE JCP3_PILOT_STOP(NOUT)
IMPLICIT NONE
INTEGER, INTENT(IN) :: NOUT
LOGICAL :: EXISTS
INQUIRE(FILE='JCP3_PILOT_ONLY',EXIST=EXISTS)
IF (EXISTS .AND. NOUT >= 1) STOP 0
END SUBROUTINE JCP3_PILOT_STOP

END MODULE JCP3_KINETIC_MOMENTS

!*****************************************************************************

"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def patch_source(source: Path, output: Path) -> dict[str, Any]:
    text = Path(source).read_text(encoding="utf-8", errors="strict")
    if "MODULE JCP3_KINETIC_MOMENTS" in text:
        raise ValueError("source already contains the JCP3 patch")
    text = replace_once(
        text,
        "PROGRAM DS2V\n!\nUSE MOLECS\n",
        MODULES + "PROGRAM DS2V\n!\nUSE MOLECS\n",
        "module insertion",
    )
    text = replace_once(
        text,
        "USE SAMPLES\nUSE MOLFILE\nUSE LJ\n\nIMPLICIT NONE\n!\nINTEGER :: IRUN",
        "USE SAMPLES\nUSE MOLFILE\nUSE JCP3_RNG_CONTROL\nUSE LJ\n\nIMPLICIT NONE\n!\nINTEGER :: IRUN",
        "RNG USE",
    )
    text = replace_once(
        text,
        "IF (IRUN == 3) WRITE (9,*) 'Starting a new run'\n",
        "IF (IRUN == 3) WRITE (9,*) 'Starting a new run'\nCALL JCP3_INIT_RANDOM_SEED(IRUN)\n",
        "RNG initialization",
    )
    text = replace_once(
        text,
        "SUBROUTINE INITIALISE_SAMPLES\n!\nUSE SAMPLES\nUSE CALC\nUSE MOLECS\nUSE STREAM\nUSE CELLS\n",
        "SUBROUTINE INITIALISE_SAMPLES\n!\nUSE SAMPLES\nUSE CALC\nUSE MOLECS\nUSE STREAM\nUSE CELLS\nUSE JCP3_KINETIC_MOMENTS\n",
        "sample reset USE",
    )
    text = replace_once(
        text,
        "CS=0. ; CSS=0. ; CSSS=0. ; CSSO=0. ; CSSOG=0. \nCO_AC=0.;",
        "CS=0. ; CSS=0. ; CSSS=0. ; CSSO=0. ; CSSOG=0.\nCALL JCP3_RESET\nCO_AC=0.;",
        "primary sample reset",
    )
    text = replace_once(
        text,
        "SUBROUTINE INITIALISE_SAMPLES101\n!\nUSE SAMPLES\nUSE CALC\nUSE MOLECS\n",
        "SUBROUTINE INITIALISE_SAMPLES101\n!\nUSE SAMPLES\nUSE CALC\nUSE MOLECS\nUSE JCP3_KINETIC_MOMENTS\n",
        "sample101 USE",
    )
    text = replace_once(
        text,
        "CS=0. ; CSS=0. ; CSSS=0. ; CSSO=0. ; CSSOG=0. \nTMF=0.\nFLF=0.\nCO_AC=0.;",
        "CS=0. ; CSS=0. ; CSSS=0. ; CSSO=0. ; CSSOG=0.\nCALL JCP3_RESET\nTMF=0.\nFLF=0.\nCO_AC=0.;",
        "secondary sample reset",
    )
    text = replace_once(
        text,
        "USE STREAM\n\n\nUSE CELLS\n!\nIMPLICIT NONE\n!\nINTEGER :: NC,NCC,LS,N,M,K",
        "USE STREAM\nUSE JCP3_KINETIC_MOMENTS\n\n\nUSE CELLS\n!\nIMPLICIT NONE\n!\nINTEGER :: NC,NCC,LS,N,M,K",
        "sample flow USE",
    )
    text = replace_once(text, "NSAMP=NSAMP+1\nWRITE (*,*) NM,'Mols. at sample',NSAMP", "NSAMP=NSAMP+1\nCALL JCP3_BEGIN_SAMPLE(NCELLS,MSP)\nWRITE (*,*) NM,'Mols. at sample',NSAMP", "sample begin")
    text = replace_once(
        text,
        "IF (IWF == 1) WF=1.+PP(2,N)*RWF\n    CS(0,NC,LS)=CS(0,NC,LS)+1.",
        "IF (IWF == 1) WF=1.+PP(2,N)*RWF\n    CALL JCP3_ACCUMULATE(NC,LS,WF,PV(1,N),PV(2,N),PV(3,N),SP(5,LS))\n    CS(0,NC,LS)=CS(0,NC,LS)+1.",
        "moment accumulation",
    )
    text = replace_once(
        text,
        "USE CHAPMAN_ENSKOG\n! \nIMPLICIT NONE",
        "USE CHAPMAN_ENSKOG\nUSE JCP3_KINETIC_MOMENTS\n! \nIMPLICIT NONE",
        "output USE",
    )
    text = replace_once(
        text,
        "CHARACTER (LEN=12) :: FS\nREAL:: DDDX,DDDT,DCX",
        "CHARACTER (LEN=12) :: FS\nCHARACTER (LEN=64) :: JCP3_WALL_FILE\nREAL:: DDDX,DDDT,DCX",
        "wall filename declaration",
    )
    text = replace_once(
        text,
        "OPEN (3,FILE='DS2SU - Copy.DAT',ERR=101)",
        "WRITE (JCP3_WALL_FILE,'(\"JCP3_WALL_NOUT\",I4.4,\".DAT\")') NOUT\n  OPEN (3,FILE=TRIM(JCP3_WALL_FILE),STATUS='REPLACE',ERR=101)",
        "wall tally filename",
    )
    text = replace_once(
        text,
        "!\nWRITE (*,*) 'Output files written'\n",
        "!\nCALL JCP3_WRITE_BLOCK(NOUT,TIME,SFAC,FNUM,NSAMP,NCELLS,MSP,CELL)\nCALL JCP3_PILOT_STOP(NOUT)\nWRITE (*,*) 'Output files written'\n",
        "block write and pilot stop",
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(text, encoding="utf-8")
    return {
        "source_sha256": sha256(source),
        "patched_source_sha256": sha256(output),
        "kinetic_moments": [
            "m0", "m1x", "m1y", "m1z", "m2xx", "m2yy", "m2zz",
            "m2xy", "m2xz", "m2yz", "energy", "energy_vx", "energy_vy",
        ],
        "wall_tally": "native DS2V surface HEAT-FLUX written per NOUT",
    }


def patch_data(source: Path, output: Path) -> dict[str, Any]:
    text = Path(source).read_text(encoding="utf-8", errors="strict")
    matches: list[tuple[int, int, str, float]] = []
    for match in FLOAT_TOKEN.finditer(text):
        token = match.group(0)
        try:
            value = float(token.replace("D", "E").replace("d", "e"))
        except ValueError:
            continue
        if math.isclose(value, M10_SPEED, rel_tol=0.0, abs_tol=0.2):
            matches.append((match.start(), match.end(), token, value))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one Mach-10 speed token, found {len(matches)}")
    start, end, old_token, old_value = matches[0]
    patched = text[:start] + f"{M12_SPEED:.8E}" + text[end:]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(patched, encoding="utf-8")
    return {
        "data_source_sha256": sha256(source),
        "patched_data_sha256": sha256(output),
        "old_speed_token": old_token,
        "old_speed_m_per_s": old_value,
        "new_speed_m_per_s": M12_SPEED,
        "speed_ratio": M12_SPEED / M10_SPEED,
        "changed_token_count": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-source", type=Path, required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "stage": "JCP3_M12_cylinder_pilot",
        "classification": "preflight_only_not_publication_evidence",
        **patch_source(args.source, args.output_source),
        **patch_data(args.data, args.output_data),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
