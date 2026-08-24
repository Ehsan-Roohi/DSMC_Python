from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "jcp11_support_validation.py"
SPEC = importlib.util.spec_from_file_location("jcp11_support_validation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class JCP11ContractTests(unittest.TestCase):
    def test_gain_contract_has_27_unique_components(self) -> None:
        self.assertEqual(len(MODULE.GAIN_KEYS), 27)
        self.assertEqual(len(set(MODULE.GAIN_KEYS)), 27)

    def test_seed_contract_is_fresh_and_disjoint(self) -> None:
        old = {
            seed
            for _, observation, reference in MODULE.M10_PAIRS
            for seed in (observation, reference)
        }
        new = set(MODULE.M12_EVALUATION_SEEDS)
        heldout = set(MODULE.M12_HELDOUT_REFERENCE_SEEDS)
        self.assertEqual(len(new), 4)
        self.assertFalse(new & old)
        self.assertFalse(new & heldout)

    def test_support_score_respects_locked_fraction(self) -> None:
        envelope = {key: 0.50 for key in MODULE.GAIN_KEYS}
        threshold = 4.0 / 27.0
        rule = {
            "gain_envelope": envelope,
            "primary_threshold_outside_fraction": threshold,
        }
        supported = {key: 0.49 for key in MODULE.GAIN_KEYS}
        shifted = dict(supported)
        for key in MODULE.GAIN_KEYS[:5]:
            shifted[key] = 0.75
        self.assertTrue(MODULE.support_score(supported, rule)["accepted"])
        self.assertFalse(MODULE.support_score(shifted, rule)["accepted"])

    def test_protocol_clarifies_block_accounting(self) -> None:
        protocol_path = (
            Path(__file__).resolve().parents[1]
            / "reference_data"
            / "mohammadzadeh_2012"
            / "jcp11_support_validation_protocol.json"
        )
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        text = protocol["block_accounting_clarifications"]["LOSO_partition"]
        self.assertIn("B3 is a subset of B10", text)
        self.assertIn("10+30=40", text)


if __name__ == "__main__":
    unittest.main()
