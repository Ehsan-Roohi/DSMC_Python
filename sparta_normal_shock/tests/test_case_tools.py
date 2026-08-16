from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate = load_module("shock_generate", "scripts/generate_case.py")
post = load_module("shock_post", "scripts/postprocess.py")


class GeneratorTests(unittest.TestCase):
    def test_mach_three_rankine_hugoniot(self):
        state = generate.rankine_hugoniot(3.0, 300.0, 1.0e25)
        self.assertAlmostEqual(state["density_ratio"], 3.0)
        self.assertAlmostEqual(state["pressure_ratio"], 11.0)
        self.assertAlmostEqual(state["temperature_ratio"], 11.0 / 3.0)
        self.assertAlmostEqual(state["velocity_2"] / state["velocity_1"], 1.0 / 3.0)

    def test_generated_contract_and_boundary_conditions(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            metadata = generate.write_case(target, "production", 3.0, 12345)
            deck = (target / "in.shock").read_text(encoding="utf-8")
            self.assertEqual(metadata["nx"], 600)
            self.assertEqual(metadata["target_upstream_particles_per_cell"], 64)
            self.assertLessEqual(metadata["transport_cfl_bound"], 0.2)
            self.assertIn("boundary             o p p", deck)
            self.assertIn("inject_right emit/face downstream xhi twopass", deck)
            self.assertIn("compute              pflux pflux/grid", deck)
            self.assertIn("compute              eflux eflux/grid", deck)


class PostprocessTests(unittest.TestCase):
    def test_parser_uses_last_snapshot(self):
        content = """ITEM: TIMESTEP
1
ITEM: NUMBER OF CELLS
1
ITEM: CELLS id xc f_avg[1] f_avg[2] f_avg[3] f_avg[4] f_avg[5] f_avg[6] f_avg[7] f_avg[8] f_avg[9] f_avg[10]
1 -1 1 2 3 4 5 6 7 8 9 10
ITEM: TIMESTEP
2
ITEM: NUMBER OF CELLS
1
ITEM: CELLS id xc f_avg[1] f_avg[2] f_avg[3] f_avg[4] f_avg[5] f_avg[6] f_avg[7] f_avg[8] f_avg[9] f_avg[10]
1 1 11 12 13 14 15 16 17 18 19 20
"""
        with tempfile.TemporaryDirectory() as temporary:
            dump = Path(temporary) / "dump"
            dump.write_text(content, encoding="utf-8")
            rows = post.read_last_grid_snapshot(dump)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["x"], 1.0)
        self.assertEqual(rows[0]["qx"], 20.0)

    def test_directional_temperature_from_pressure_tensor(self):
        n1 = 1.0e20
        t1 = 300.0
        base = generate.rankine_hugoniot(3.0, t1, n1)
        meta = {
            **base,
            "argon_mass_kg": generate.MASS,
            "mean_free_path_1": 1.0,
            "half_span_lambda": 10.0,
        }
        rows = []
        for index in range(20):
            downstream = index >= 10
            n = base["number_density_2"] if downstream else n1
            temp = base["temperature_2"] if downstream else t1
            velocity = base["velocity_2"] if downstream else base["velocity_1"]
            pressure = n * post.K_B * temp
            rows.append({
                "x": -9.5 + index,
                "number_density": n,
                "u": velocity,
                "v": 0.0,
                "w": 0.0,
                "temperature": temp,
                "pressure": pressure,
                "Pxx": pressure,
                "Pyy": pressure,
                "Pzz": pressure,
                "qx": 0.0,
            })
        normalized, metrics = post.normalize_rows(rows, meta)
        self.assertAlmostEqual(normalized[0]["Tx_over_T1"], 1.0)
        self.assertAlmostEqual(normalized[-1]["Tperp_over_T1"], 11.0 / 3.0)
        self.assertLess(metrics["gates"]["far_field_max_relative_error"], 1.0e-12)


if __name__ == "__main__":
    unittest.main()
