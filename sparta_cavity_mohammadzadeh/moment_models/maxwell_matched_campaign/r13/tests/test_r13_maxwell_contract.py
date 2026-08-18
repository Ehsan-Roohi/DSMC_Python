#!/usr/bin/env python3
"""Fail-closed contract tests for the Maxwell R13 production source."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
sys.path.insert(0, str(SOURCE))

import r13_maxwell_production as maxwell  # noqa: E402
import rana_original_coefficients as coefficients  # noqa: E402
import rana_original_reference_solver as solver  # noqa: E402


def main() -> int:
    assert tuple(solver.STATE_ORDER) == maxwell.STATE_ORDER_SOLVER
    assert solver.flux_x(np.ones(17))[7, 13] == 1.0
    state = np.asarray(
        [
            1.1, 0.02, -0.03, 1.04, 0.01, -0.012, 0.02, -0.008,
            -0.013, 0.001, -0.0015, 0.002, 0.0004, -0.0005,
            0.0006, -0.0007, 0.0008,
        ],
        dtype=float,
    )
    for kn_gu in (0.05, 0.20):
        kn_rana = maxwell.kn_rana_from_gu(kn_gu)
        actual = coefficients.production(state, kn_rana, rb=1.0, ra=1.0, ma=1.0)
        expected = maxwell.production_maxwell(state, kn_rana)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        assert abs(maxwell.kn_gu_from_rana(kn_rana) - kn_gu) < 2.0e-16
    print("R13_MAXWELL_CONTRACT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
