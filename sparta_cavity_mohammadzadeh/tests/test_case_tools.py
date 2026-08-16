from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CaseToolTests(unittest.TestCase):
    def generate(self, level: str) -> tuple[str, dict[str, object]]:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "case"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "generate_case.py"),
                    "--level",
                    level,
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            deck = (output / "in.cavity").read_text(encoding="utf-8")
            metadata = json.loads((output / "case_metadata.json").read_text(encoding="utf-8"))
        return deck, metadata

    def test_smoke_deck_has_expected_physics(self) -> None:
        deck, metadata = self.generate("smoke")
        self.assertIn("boundary             s s p", deck)
        self.assertIn("translate 100", deck)
        self.assertIn("collide              vss gas argon.vss", deck)
        self.assertAlmostEqual(metadata["kn"], 0.1)
        self.assertLess(metadata["dt_over_collision_time"], 0.1)

    def test_temperature_is_com_subtracted_thermal_temperature(self) -> None:
        deck, metadata = self.generate("smoke")
        self.assertIn("compute              flow grid all gas nrho u v w\n", deck)
        self.assertNotIn("compute              flow grid all gas nrho u v w temp", deck)
        self.assertIn("compute              thermal thermal/grid all gas temp", deck)
        self.assertIn("c_flow[*] c_thermal[*] ave running", deck)
        self.assertEqual(
            metadata["temperature_observable"],
            "thermal/grid COM-subtracted translational temperature",
        )
        self.assertEqual(
            metadata["dump_columns"],
            ["nrho", "u", "v", "w", "thermal_temperature"],
        )

    def test_hq_preset_increases_independent_statistics(self) -> None:
        deck, metadata = self.generate("hq")
        self.assertEqual(metadata["nx"], 200)
        self.assertEqual(metadata["ny"], 200)
        self.assertEqual(metadata["particles_per_cell"], 128)
        self.assertEqual(metadata["nparticles"], 5_120_000)
        self.assertEqual(metadata["warmup_steps"], 40_000)
        self.assertEqual(metadata["sample_steps"], 160_000)
        self.assertEqual(metadata["sample_stride"], 10)
        self.assertIn("run                  40000", deck)
        self.assertIn("dump                 fields grid all 160000", deck)
        self.assertIn("run                  160000", deck)


if __name__ == "__main__":
    unittest.main()
