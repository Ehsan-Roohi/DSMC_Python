from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


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
            self.assertLess(metadata["dt_over_collision_time"], 0.1)


if __name__ == "__main__":
    unittest.main()
