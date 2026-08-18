#!/usr/bin/env python3
"""Public, source-independent checks for the printed R13 production operator."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import sys
import unittest

import numpy as np


EQUATIONS = Path(__file__).resolve().parents[1] / "equations"
sys.path.insert(0, str(EQUATIONS))

from r13_maxwell_production import (  # noqa: E402
    NVAR,
    STATE_ORDER_PRINTED_EQ11,
    STATE_ORDER_SOLVER,
    appendix_a_reduced_production_matrix,
    kn_gu_from_rana,
    kn_rana_from_gu,
    maxwell_collision_prefactor,
    production_appendix_literal,
    production_maxwell,
)


class R13EquationContract(unittest.TestCase):
    def setUp(self) -> None:
        self.state = np.array(
            [
                1.2,
                0.04,
                -0.03,
                0.9,
                0.06,
                -0.02,
                0.03,
                -0.01,
                -0.02,
                0.015,
                -0.004,
                -0.011,
                0.002,
                -0.001,
                0.003,
                -0.004,
                0.02,
            ],
            dtype=float,
        )

    def test_state_order_is_explicit(self) -> None:
        self.assertEqual(NVAR, 17)
        self.assertEqual(len(STATE_ORDER_SOLVER), NVAR)
        self.assertEqual(len(STATE_ORDER_PRINTED_EQ11), NVAR)
        self.assertEqual(STATE_ORDER_SOLVER[13:15], ("m_xxy", "m_xyy"))
        self.assertEqual(STATE_ORDER_PRINTED_EQ11[13:15], ("m_xyy", "m_xxy"))

    def test_exact_diagonal_relaxation_coefficients(self) -> None:
        state = [Fraction(str(value)) for value in self.state]
        matrix = appendix_a_reduced_production_matrix(state, exact=True)
        self.assertEqual(matrix.shape, (NVAR, NVAR))
        expected = {
            4: Fraction(2, 3),
            5: Fraction(2, 3),
            6: Fraction(1),
            7: Fraction(1),
            8: Fraction(1),
            9: Fraction(5, 24),
            10: Fraction(5, 24),
            11: Fraction(5, 24),
            12: Fraction(1, 2),
            13: Fraction(1, 2),
            14: Fraction(1, 2),
            15: Fraction(1, 2),
            16: Fraction(2, 3),
        }
        for index, value in expected.items():
            self.assertEqual(matrix[index, index], value)
        self.assertEqual(matrix[0, 0], Fraction(0))
        self.assertEqual(matrix[3, 3], Fraction(0))

    def test_maxwell_prefactor_and_literal_branch_are_distinct(self) -> None:
        kn_rana = 0.12
        matrix = appendix_a_reduced_production_matrix(self.state)
        np.testing.assert_allclose(
            production_maxwell(self.state, kn_rana),
            self.state[0] * matrix / kn_rana,
            rtol=0.0,
            atol=2e-15,
        )
        np.testing.assert_allclose(
            production_appendix_literal(self.state, kn_rana),
            matrix / kn_rana,
            rtol=0.0,
            atol=2e-15,
        )
        self.assertEqual(
            maxwell_collision_prefactor(self.state, kn_rana),
            self.state[0] / kn_rana,
        )

    def test_knudsen_conversion_round_trip(self) -> None:
        for kn_gu in (0.05, 0.20):
            self.assertAlmostEqual(kn_gu_from_rana(kn_rana_from_gu(kn_gu)), kn_gu)

    def test_invalid_state_and_knudsen_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            appendix_a_reduced_production_matrix(self.state[:-1])
        with self.assertRaises(ValueError):
            production_maxwell(self.state, 0.0)
        invalid = self.state.copy()
        invalid[0] = -1.0
        with self.assertRaises(FloatingPointError):
            production_maxwell(invalid, 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
