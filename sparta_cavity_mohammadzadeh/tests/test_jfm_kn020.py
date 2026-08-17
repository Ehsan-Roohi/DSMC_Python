from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_jfm_kn020_case.py"
SPEC = importlib.util.spec_from_file_location("generate_jfm_kn020_case", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

POST_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "postprocess_jfm_kn020.py"
POST_SPEC = importlib.util.spec_from_file_location("postprocess_jfm_kn020", POST_SCRIPT)
assert POST_SPEC and POST_SPEC.loader
POST = importlib.util.module_from_spec(POST_SPEC)
POST_SPEC.loader.exec_module(POST)


class JFMKn020CaseTests(unittest.TestCase):
    def test_vhs_mean_free_path_contract(self) -> None:
        values = MODULE.physical_parameters(0.20, 1.0e-6, 160, 128, 300.0)
        n = values["number_density_m-3"]
        factor = (MODULE.TEMPERATURE_REF / 300.0) ** (MODULE.VISCOSITY_INDEX - 0.5)
        reconstructed = 1.0 / (
            math.sqrt(2.0) * math.pi * MODULE.DIAMETER_REF**2 * n * factor
        )
        self.assertAlmostEqual(reconstructed / 1.0e-6, 0.20, places=13)

    def test_publication_deck_has_temperature_and_heat_flux(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "case"
            metadata = MODULE.write_case(output, seed=20260803)
            deck = (output / "in.cavity").read_text(encoding="utf-8")
            self.assertIn("thermal/grid all gas temp", deck)
            self.assertIn("eflux/grid all gas heatx heaty", deck)
            self.assertIn("c_flow[*] c_thermal[*] c_heat[*] ave running", deck)
            self.assertEqual(metadata["accumulated_samples_per_cell"], 8501)
            self.assertEqual(metadata["dump_columns"], ["nrho", "u", "v", "w", "T", "qx", "qy"])
            self.assertEqual(metadata["argon_mass_kg"], 6.6335e-26)
            on_disk = json.loads((output / "case_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(on_disk["kn_convention"], "gu_lambda_over_L")

    def test_dump_reader_and_student_t_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dump = Path(tmp) / "grid.final.00085010"
            dump.write_text(
                "ITEM: TIMESTEP\n85010\n"
                "ITEM: NUMBER OF CELLS\n2\n"
                "ITEM: CELLS id xc yc f_fieldavg[1] f_fieldavg[2] "
                "f_fieldavg[3] f_fieldavg[4] f_fieldavg[5] f_fieldavg[6] f_fieldavg[7]\n"
                "1 0.25 0.25 10 1 2 0 300 4 5\n"
                "2 0.75 0.25 11 2 3 0 301 5 6\n",
                encoding="utf-8",
            )
            columns, values = POST.read_last_snapshot(dump)
            self.assertEqual(columns[-1], "f_fieldavg[7]")
            self.assertEqual(values.shape, (2, 10))
            samples = POST.np.arange(16.0).reshape(8, 2)
            interval = POST.ci95(samples)
            expected = POST.T95[8] * POST.np.std(samples, axis=0, ddof=1) / math.sqrt(8)
            self.assertTrue(POST.np.allclose(interval, expected))


if __name__ == "__main__":
    unittest.main()
