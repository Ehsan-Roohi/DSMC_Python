from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PAYLOAD = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PATCHER = load_module("mv11_patcher", PAYLOAD / "patcher" / "patch_ds2v_mv11.py")
ANALYSIS = load_module("mv11_analysis", PAYLOAD / "tools" / "analyze_mv11_cylinder.py")


def miniature_source() -> str:
    return """PROGRAM DS2V
USE MOLFILE
IMPLICIT NONE
INTEGER :: IRUN
IF (IRUN == 3) WRITE (9,*) 'Starting a new run'
CALL WRITE_RESTART(0)
CALL WRITE_RESTART(1)
END PROGRAM DS2V

SUBROUTINE SAMPLE_FLOW
USE SAMPLES
IMPLICIT NONE
INTEGER :: N,NC,LS,NCELLS,MSP,IWF
REAL :: WF,RWF
REAL :: PP(2,1),PV(3,1),SP(5,1)
NSAMP=NSAMP+1
IF (IWF == 1) WF=1.+PP(2,N)*RWF
END SUBROUTINE SAMPLE_FLOW

SUBROUTINE OUTPUT_RESULTS
USE SAMPLES
IMPLICIT NONE
WRITE (*,*) 'Output files written'
END SUBROUTINE OUTPUT_RESULTS

SUBROUTINE INITIALISE_SAMPLES
USE SAMPLES
CS=0. ; CSS=0. ; CSSS=0. ; CSSO=0. ; CSSOG=0.
END SUBROUTINE INITIALISE_SAMPLES

SUBROUTINE INITIALISE_SAMPLES101
USE SAMPLES
CS=0. ; CSS=0. ; CSSS=0. ; CSSO=0. ; CSSOG=0.
END SUBROUTINE INITIALISE_SAMPLES101

SUBROUTINE WRITE_RESTART(IST)
INTEGER :: IST
END SUBROUTINE WRITE_RESTART

SUBROUTINE RANDOM_DRAW(RANF,X)
REAL :: RANF,X
X=-LOG(RANF)
END SUBROUTINE RANDOM_DRAW
"""


def test_patcher_is_fail_closed_and_inserts_required_markers(tmp_path: Path):
    source = tmp_path / "base.F90"
    output = tmp_path / "mv11.F90"
    report = tmp_path / "report.json"
    source.write_text(miniature_source())
    PATCHER.patch_source(source, output, report)
    text = output.read_text()
    assert "MODULE RNG_CONTROL" in text
    assert "MODULE MV11_KINETIC_MOMENTS" in text
    assert "CALL INIT_RANDOM_SEED(IRUN)" in text
    assert text.count("CALL SAVE_RANDOM_STATE") == 2
    assert "LOG(MAX(RANF,0.5*EPSILON(RANF)))" in text
    assert "CALL MV11_ACCUMULATE" in text
    assert "CALL MV11_WRITE_BLOCK" in text
    assert "REAL(KIND=8), INTENT(IN) :: TIME" in text
    assert "REAL, INTENT(IN) :: SFAC,FNUM" in text
    assert "REAL, INTENT(IN) :: TIME,SFAC,FNUM" not in text
    # Two DS2V reset hooks plus the reset after each written MV11 block.
    assert text.count("CALL MV11_RESET") == 3


def test_reconstruction_matches_direct_central_moment():
    mass = 2.5
    velocity = np.array(
        [[2.0, -1.0, 0.5], [4.0, 3.0, -0.5], [-1.0, 2.0, 1.5], [3.0, 0.0, -2.0]]
    )
    weight = np.array([1.0, 2.0, 1.5, 0.5])
    energy = 0.5 * mass * np.sum(velocity * velocity, axis=1)
    moments = np.array(
        [
            np.sum(weight),
            np.sum(weight * velocity[:, 0]),
            np.sum(weight * velocity[:, 1]),
            np.sum(weight * velocity[:, 2]),
            np.sum(weight * velocity[:, 0] ** 2),
            np.sum(weight * velocity[:, 1] ** 2),
            np.sum(weight * velocity[:, 2] ** 2),
            np.sum(weight * velocity[:, 0] * velocity[:, 1]),
            np.sum(weight * velocity[:, 0] * velocity[:, 2]),
            np.sum(weight * velocity[:, 1] * velocity[:, 2]),
            np.sum(weight * energy),
            np.sum(weight * energy * velocity[:, 0]),
            np.sum(weight * energy * velocity[:, 1]),
        ]
    )
    raw = np.concatenate(([1.0, 1.0, 0.25, 0.5, 2.0], moments))[None, :]
    metadata = {"NOUT": 1.0, "TIME": 1.0, "FNUM": 3.0, "BLOCK_SAMPLES": 2.0}
    result = ANALYSIS.reconstruct(metadata, raw, mass)

    mean_velocity = np.sum(weight[:, None] * velocity, axis=0) / np.sum(weight)
    peculiar = velocity - mean_velocity
    number_density = 3.0 * np.sum(weight) / (2.0 * 2.0)
    direct_pxy = number_density * mass * np.sum(weight * peculiar[:, 0] * peculiar[:, 1]) / np.sum(weight)
    direct_qx = number_density * np.sum(
        weight * 0.5 * mass * np.sum(peculiar * peculiar, axis=1) * peculiar[:, 0]
    ) / np.sum(weight)
    direct_qy = number_density * np.sum(
        weight * 0.5 * mass * np.sum(peculiar * peculiar, axis=1) * peculiar[:, 1]
    ) / np.sum(weight)
    assert np.isclose(result["Pxy_Pa"][0], direct_pxy)
    assert np.isclose(result["qx_W_m2"][0], direct_qx)
    assert np.isclose(result["qy_W_m2"][0], direct_qy)


def test_protocol_uses_fresh_unique_seeds():
    import json

    protocol = json.loads((PAYLOAD / "case" / "mv11_cylinder_protocol.json").read_text())
    seeds = protocol["prospectively_locked_fresh_seeds"]
    forbidden = set(protocol["forbidden_legacy_seeds"])
    assert len(seeds) == 4
    assert len(set(seeds)) == 4
    assert not (set(seeds) & forbidden)


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        test_patcher_is_fail_closed_and_inserts_required_markers(Path(directory))
    test_reconstruction_matches_direct_central_moment()
    test_protocol_uses_fresh_unique_seeds()
    print("MV11_TESTS_PASS count=3")


if __name__ == "__main__":
    main()
