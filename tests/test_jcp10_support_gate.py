from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "jcp10_support_gate.py"
SPEC = importlib.util.spec_from_file_location("jcp10_support_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _development(limit: float = 0.20) -> dict[str, np.ndarray]:
    return {
        f"gain_{component}_{zone}": np.asarray([0.05, 0.10, limit])
        for component in MODULE.GAIN_COMPONENTS
        for zone in MODULE.ZONES
    }


def _unit(seed: int, gain: float) -> dict[str, object]:
    return {
        "seed": seed,
        "gains": {
            component: [gain, gain, gain]
            for component in MODULE.GAIN_COMPONENTS
        },
    }


class SupportGateTests(unittest.TestCase):
    def test_support_rows_accept_inside_and_flag_outside(self) -> None:
        manifest = {"units": [_unit(1, 0.19), _unit(2, 0.21)]}
        rows = MODULE.support_rows(_development(), manifest)
        self.assertEqual(len(rows), 54)
        by_seed = {
            seed: [
                row["outside_development_support"]
                for row in rows
                if row["seed"] == seed
            ]
            for seed in (1, 2)
        }
        self.assertFalse(any(by_seed[1]))
        self.assertTrue(all(by_seed[2]))

    def test_nrmse_and_geometric_mean_are_unit_consistent(self) -> None:
        target = np.asarray([[1.0, 2.0], [1.0, 2.0]])
        value = np.asarray([[2.0, 4.0], [2.0, 4.0]])
        errors = MODULE.nrmse(value, target)
        self.assertTrue(np.allclose(errors, [1.0, 1.0]))
        self.assertTrue(np.isclose(MODULE.geometric(errors), 1.0))


if __name__ == "__main__":
    unittest.main()
