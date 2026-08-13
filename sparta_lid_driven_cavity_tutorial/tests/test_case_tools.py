from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from postprocess import moving_average  # noqa: E402


class CaseToolTests(unittest.TestCase):
    def test_smoke_deck_has_expected_physics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "case"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "generate_case.py"),
                    "--level",
                    "smoke",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            deck = (output / "in.cavity").read_text(encoding="utf-8")
            metadata = json.loads((output / "case_metadata.json").read_text(encoding="utf-8"))
            self.assertIn("boundary             s s p", deck)
            self.assertIn("translate 100", deck)
            self.assertIn("collide              vss gas argon.vss", deck)
            self.assertAlmostEqual(metadata["kn"], 0.1)
            self.assertEqual(metadata["vhs_alpha"], 1.0)
            self.assertLess(metadata["dt_over_collision_time"], 0.1)

    def test_production_resolution_and_population(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "case"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "generate_case.py"),
                    "--level",
                    "production",
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            metadata = json.loads((output / "case_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["nx"], 200)
            self.assertEqual(metadata["particles_per_cell"], 32)
            self.assertEqual(metadata["nparticles"], 1_280_000)
            self.assertAlmostEqual(metadata["dx_over_lambda"], 0.05)

    def test_moving_average_preserves_constant_data(self) -> None:
        data = np.full(20, 7.5)
        np.testing.assert_allclose(moving_average(data, 11), data)


if __name__ == "__main__":
    unittest.main()
