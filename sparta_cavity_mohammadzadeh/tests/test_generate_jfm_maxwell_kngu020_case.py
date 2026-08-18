import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "generate_jfm_maxwell_kngu020_case.py"
SPEC = importlib.util.spec_from_file_location("maxwell_case", MODULE_PATH)
assert SPEC and SPEC.loader
maxwell_case = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(maxwell_case)


class MaxwellKnGuCaseTests(unittest.TestCase):
    def test_transport_contract_at_both_publication_knudsen_numbers(self) -> None:
        expected = {
            0.05: (3.4538410244383913e25, 5.270143164731432e6, 5.0e-8),
            0.20: (8.634602561095979e24, 1.317535791182858e6, 2.0e-7),
        }
        for kn_gu, (density, fnum, mean_free_path) in expected.items():
            with self.subTest(kn_gu=kn_gu):
                values = maxwell_case.physical_parameters(
                    kn_gu, 1.0e-6, 160, 256, 300.0
                )
                self.assertTrue(
                    math.isclose(values["number_density_m-3"], density, rel_tol=2e-15)
                )
                self.assertTrue(math.isclose(values["fnum"], fnum, rel_tol=2e-15))
                self.assertTrue(
                    math.isclose(
                        values["mean_free_path_gu_m"], mean_free_path, rel_tol=2e-15
                    )
                )
                self.assertTrue(
                    math.isclose(values["kn_gu_reconstructed"], kn_gu, rel_tol=2e-15)
                )
        self.assertTrue(
            math.isclose(
                maxwell_case.DIAMETER_VHS_EQUIVALENT,
                4.632665904220862e-10,
                rel_tol=2e-15,
            )
        )
        self.assertTrue(
            math.isclose(
                maxwell_case.DIAMETER_VSS_INPUT,
                4.661368788251916e-10,
                rel_tol=2e-15,
            )
        )

    def test_vss_and_viscosity_equivalent_diameters_are_distinct(self) -> None:
        self.assertGreater(
            maxwell_case.DIAMETER_VSS_INPUT,
            maxwell_case.DIAMETER_VHS_EQUIVALENT,
        )
        self.assertTrue(
            math.isclose(
                (maxwell_case.DIAMETER_VSS_INPUT / maxwell_case.DIAMETER_VHS_EQUIVALENT)
                ** 2,
                maxwell_case.VSS_AREA_FACTOR,
                rel_tol=2e-15,
            )
        )

    def test_written_case_is_self_consistent_and_has_antifourier_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "case"
            maxwell_case.write_case(
                output,
                seed=104729,
                kn_gu=0.05,
                nx=12,
                ppc=4,
                warmup_steps=20,
                sample_steps=30,
                sample_stride=2,
                checkpoint_steps=0,
            )
            metadata = json.loads((output / "case_metadata.json").read_text())
            self.assertEqual(metadata["kn_convention"], "gu_lambda_over_L")
            self.assertEqual(metadata["viscosity_index"], 1.0)
            self.assertEqual(metadata["vss_alpha"], 2.14)
            self.assertEqual(metadata["evidence_level"], "single_realisation_model_audit")
            self.assertEqual(metadata["dump_field_count"], 15)
            self.assertEqual(metadata["dump_columns"], maxwell_case.DUMP_COLUMNS)
            self.assertFalse(
                metadata["moment_sampling"]["direct_rank3_moment_m_ijk_available"]
            )
            self.assertFalse(
                metadata["moment_sampling"]["full_r26_higher_moment_claim"]
            )
            self.assertTrue(
                metadata["moment_sampling"]["instantaneous_COM_sonine"]
            )
            self.assertEqual(
                metadata["moment_sampling"]["sonine_role"], "diagnostic_only"
            )
            self.assertFalse(
                metadata["moment_sampling"]["quantitative_R_or_Delta_claim_ready"]
            )
            deck = (output / "in.cavity").read_text()
            self.assertIn("collide              vss gas maxwell.vss", deck)
            self.assertIn(
                "compute              stress pflux/grid all gas momxx momxy momyy momzz",
                deck,
            )
            self.assertIn(
                "compute              sonine sonine/grid all gas b xx 1 b xy 1 b yy 1 b zz 1",
                deck,
            )
            self.assertIn("c_stress[*] c_sonine[*]", deck)
            self.assertIn(" 1 273 2.14", (output / "maxwell.vss").read_text())

    def test_default_campaign_resolution_and_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for index, kn_gu in enumerate((0.05, 0.20)):
                output = Path(temporary) / f"case_{index}"
                metadata = maxwell_case.write_case(
                    output,
                    seed=104729,
                    kn_gu=kn_gu,
                )
                self.assertEqual(metadata["nx"], 160)
                self.assertEqual(metadata["particles_per_cell"], 256)
                self.assertEqual(metadata["warmup_steps"], 40_000)
                self.assertEqual(metadata["sample_steps"], 200_000)
                self.assertEqual(metadata["accumulated_samples_per_cell"], 20_000)
                self.assertTrue(
                    math.isclose(metadata["kn_gu_reconstructed"], kn_gu, rel_tol=2e-15)
                )


if __name__ == "__main__":
    unittest.main()
