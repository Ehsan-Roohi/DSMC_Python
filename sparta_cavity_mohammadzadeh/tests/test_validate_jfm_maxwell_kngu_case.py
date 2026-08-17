import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = load_module(
    "maxwell_case_for_validator",
    ROOT / "scripts" / "generate_jfm_maxwell_kngu020_case.py",
)
validator = load_module(
    "maxwell_validator",
    ROOT / "scripts" / "validate_jfm_maxwell_kngu_case.py",
)


class MaxwellCaseValidatorTests(unittest.TestCase):
    def test_generated_contract_and_completed_15_field_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case_dir = Path(temporary) / "kn020"
            generator.write_case(case_dir, seed=104729, kn_gu=0.20)
            self.assertIsNone(validator.validate_case(case_dir, 0.20, False))

            fields = " ".join(f"f_fieldavg[{index}]" for index in range(1, 16))
            final = case_dir / "grid.final.00200000"
            final.write_text(
                "ITEM: TIMESTEP\n200000\n"
                "ITEM: NUMBER OF CELLS\n1\n"
                "ITEM: BOX BOUNDS ss ss pp\n0 1\n0 1\n-0.5 0.5\n"
                f"ITEM: CELLS id xc yc {fields}\n"
                f"1 0.5 0.5 {' '.join('0' for _ in range(15))}\n",
                encoding="utf-8",
            )
            self.assertEqual(validator.validate_case(case_dir, 0.20, True), final)

    def test_wrong_kn_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case_dir = Path(temporary) / "kn005"
            generator.write_case(case_dir, seed=104729, kn_gu=0.05)
            with self.assertRaisesRegex(ValueError, "kn_gu"):
                validator.validate_case(case_dir, 0.20, False)


if __name__ == "__main__":
    unittest.main()
